"""
Feature extraction de tracking para el clasificador de eventos (fase 3).

Reutiliza el pipeline existente (lib.Match + EventGenerator) para detectar y
filtrar posesiones, y convierte cada TRANSICION entre posesiones consecutivas
en una fila de features. La transicion es la unidad de muestra: es exactamente
donde las reglas deciden PASS / BALL LOST / SHOT / etc., asi que el modelo
aprende sobre los mismos candidatos que hoy clasifican las reglas.

Uso (convencion por partido, resuelve las rutas solo):
    python3 events_model/features.py --match psg_bayern

Uso (rutas explicitas):
    python3 events_model/features.py \
        --tracking ~/football_data/matches/psg_bayern/tracking.csv \
        --match-id psg_bayern \
        --labels events_model/dataset/psg_bayern_labeled.csv \
        --out events_model/dataset/psg_bayern_features.csv

Nota: build_dataset.py ya llama a extract() internamente para cada --match,
asi que normalmente no hace falta correr este script a mano; sirve para
inspeccionar las features de un partido suelto o generarlas sin labels
(inferencia).

--labels es opcional: sin labels genera solo las features (para inferencia).
Con labels (el ground truth del label_tool) agrega la columna `label`.

Todas las posiciones se normalizan a [0,1] y se ORIENTAN: x=1 es siempre el
arco que ataca el equipo de la posesion actual. Sin esto el modelo aprende
"de que lado juega cada equipo en este clip" y no generaliza entre partidos.
"""

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_cleanup"))

import matchpaths                    # noqa: E402
from lib.match import Match          # noqa: E402
from lib.event_generator import EventGenerator  # noqa: E402
from lib import pitch                # noqa: E402

# Al unir labels: un evento de ground truth pertenece a una transicion si su
# Start Frame cae dentro de la ventana de la transicion +- esta tolerancia.
LABEL_FRAME_TOLERANCE = 12

# Ventana (en transiciones) para el feature de ping-pong de posesiones.
PINGPONG_WINDOW = 6

FIELDNAMES = [
    "match_id", "start_frame", "end_frame",
    # posesiones
    "dur_prev", "dur_curr", "dur_nxt", "touches_curr",
    "same_team", "same_player",
    # vuelo de la pelota en el gap
    "gap_frames", "ball_det_rate", "max_blank_run",
    "net_disp_m", "path_len_m", "straightness", "mean_speed_ms",
    # geometria (orientada: arco atacado = x=1)
    "start_x", "start_y", "end_x", "end_y",
    "dist_goal_start_m", "dist_goal_end_m", "delta_goal_dist_m",
    "lateral_disp_m", "goalward",
    # contexto de jugadores
    "opp_dist_release_m", "opp_dist_receive_m",
    "density_5m_release", "density_10m_release",
    "density_5m_receive", "density_10m_receive",
    # secuencia
    "team_flips_recent",
    # etiqueta (vacia si no hay ground truth)
    "label",
]


def orient(loc, attacking_goal):
    """Refleja un punto normalizado para que el arco atacado quede en x=1."""
    if loc is None:
        return None
    x, y = loc
    return (x, y) if attacking_goal == 1.0 else (1.0 - x, 1.0 - y)


def player_context(gen, frame_number, ball_norm, team):
    """(dist al rival mas cercano de la pelota, densidad a 5m, a 10m)."""
    moment = gen.match.frame(frame_number)
    if moment is None or ball_norm is None:
        return None, 0, 0
    opp_dist = None
    d5 = d10 = 0
    for entry in moment.players:
        fr = entry["frame"]
        if fr is None or fr.coordinates is None:
            continue
        p_norm = pitch.to_normalised(gen.source, fr.x, fr.y)
        d = pitch.distance_m(ball_norm, p_norm)
        if d <= 5:
            d5 += 1
        if d <= 10:
            d10 += 1
        if str(entry["object"].team) != str(team):
            if opp_dist is None or d < opp_dist:
                opp_dist = d
    return opp_dist, d5, d10


def gap_ball_path(gen, a, b):
    """Posiciones (frame, norm) de la pelota entre dos posesiones + blanks."""
    path = []
    max_blank = blank = 0
    frames_total = 0
    for fn in range(a.end_frame, b.start_frame + 1):
        moment = gen.match.frame(fn)
        if moment is None:
            continue
        frames_total += 1
        ball = gen._norm_ball(moment)
        if ball is None:
            blank += 1
            max_blank = max(max_blank, blank)
            continue
        blank = 0
        path.append((fn, ball))
    return path, frames_total, max_blank


def transition_features(gen, prev, curr, nxt, recent_teams, fps):
    """Una fila de features para la transicion curr -> nxt."""
    goal_x = gen._attacking_goal(curr.team)
    goal = (1.0, 0.5)  # tras orientar, el arco atacado es siempre (1, .5)

    path, frames_total, max_blank = gap_ball_path(gen, curr, nxt)
    start = orient(curr.end_loc, goal_x)
    end = orient(nxt.start_loc, goal_x)

    path_len = 0.0
    pts = [orient(p, goal_x) for _, p in path]
    for i in range(1, len(pts)):
        path_len += pitch.distance_m(pts[i - 1], pts[i])
    net = pitch.distance_m(start, end) if start and end else 0.0
    gap_frames = max(nxt.start_frame - curr.end_frame - 1, 0)
    gap_seconds = max(gap_frames, 1) / fps

    d_start = pitch.distance_m(start, goal) if start else ""
    d_end = pitch.distance_m(end, goal) if end else ""
    delta_goal = (d_start - d_end) if start and end else ""

    opp_rel, d5_rel, d10_rel = player_context(
        gen, curr.end_frame, curr.end_loc, curr.team)
    opp_rec, d5_rec, d10_rec = player_context(
        gen, nxt.start_frame, nxt.start_loc, nxt.team)

    flips = sum(1 for i in range(1, len(recent_teams))
                if recent_teams[i] != recent_teams[i - 1])

    return {
        "start_frame": curr.end_frame,
        "end_frame": nxt.start_frame,
        "dur_prev": prev.duration if prev else 0,
        "dur_curr": curr.duration,
        "dur_nxt": nxt.duration,
        "touches_curr": curr.touches,
        "same_team": int(curr.same_team(nxt)),
        "same_player": int(curr.same_player(nxt)),
        "gap_frames": gap_frames,
        "ball_det_rate": round(len(path) / frames_total, 3) if frames_total else 0,
        "max_blank_run": max_blank,
        "net_disp_m": round(net, 2),
        "path_len_m": round(path_len, 2),
        "straightness": round(net / path_len, 3) if path_len > 0 else 1.0,
        "mean_speed_ms": round(net / gap_seconds, 2),
        "start_x": round(start[0], 4) if start else "",
        "start_y": round(start[1], 4) if start else "",
        "end_x": round(end[0], 4) if end else "",
        "end_y": round(end[1], 4) if end else "",
        "dist_goal_start_m": round(d_start, 2) if d_start != "" else "",
        "dist_goal_end_m": round(d_end, 2) if d_end != "" else "",
        "delta_goal_dist_m": round(delta_goal, 2) if delta_goal != "" else "",
        "lateral_disp_m": round(abs(start[1] - end[1]) * pitch.PITCH_WIDTH_M, 2)
                          if start and end else "",
        "goalward": int(gen._is_goalward(curr.end_loc, nxt.start_loc, goal_x)),
        "opp_dist_release_m": round(opp_rel, 2) if opp_rel is not None else "",
        "opp_dist_receive_m": round(opp_rec, 2) if opp_rec is not None else "",
        "density_5m_release": d5_rel,
        "density_10m_release": d10_rel,
        "density_5m_receive": d5_rec,
        "density_10m_receive": d10_rec,
        "team_flips_recent": flips,
        "label": "",
    }


def load_labels(path):
    """Ground truth del label_tool: [(start_frame, tipo)] sin los REJECTED."""
    labels = []
    with open(os.path.expanduser(path), newline="") as f:
        for row in csv.DictReader(f):
            if row.get("Verdict") == "REJECTED":
                continue
            labels.append((int(float(row["Start Frame"])), row["Type"]))
    return labels


def attach_labels(rows, labels):
    """Asigna a cada transicion el evento de GT mas cercano dentro de su ventana.

    Transiciones sin evento cercano quedan como NONE: son los negativos (los
    micro-cambios de posesion que las reglas hoy cuentan de mas).
    """
    for row in rows:
        lo = row["start_frame"] - LABEL_FRAME_TOLERANCE
        hi = row["end_frame"] + LABEL_FRAME_TOLERANCE
        best, best_dist = "NONE", None
        for frame, etype in labels:
            if lo <= frame <= hi:
                dist = min(abs(frame - row["start_frame"]),
                           abs(frame - row["end_frame"]))
                if best_dist is None or dist < best_dist:
                    best, best_dist = etype, dist
        row["label"] = best
    return rows


def extract(tracking_csv, match_id, labels_csv=None):
    tracking_csv = os.path.expanduser(tracking_csv)
    match = Match()
    match.import_raw_data(os.path.dirname(tracking_csv) + os.sep,
                          os.path.basename(tracking_csv))
    gen = EventGenerator(match)
    gen.possessions = gen.filter_possessions(gen.detect_possessions())
    gen._infer_attacking_directions()
    fps = gen.match_fps()

    rows = []
    poss = gen.possessions
    for i in range(len(poss) - 1):
        prev = poss[i - 1] if i > 0 else None
        recent = [str(p.team) for p in poss[max(0, i - PINGPONG_WINDOW):i + 1]]
        row = transition_features(gen, prev, poss[i], poss[i + 1], recent, fps)
        row["match_id"] = match_id
        rows.append(row)

    if labels_csv:
        rows = attach_labels(rows, load_labels(labels_csv))

    # TODO(fase 3): features de SET PIECE (posicion del restart tras blank largo)
    # TODO(fase 3): features de DUEL (dos rivales cerca de la pelota en el gap)
    # TODO(fase 3): sumar el score del spotter de video (T-DEED) como feature
    return rows


def main():
    ap = argparse.ArgumentParser(description="Features de tracking por transicion")
    ap.add_argument("--match", default=None,
                    help="nombre del partido (convencion "
                         "~/football_data/matches/<match>/ + events_model/dataset/)")
    ap.add_argument("--tracking", default=None, help="CSV crudo de tracking (override de --match)")
    ap.add_argument("--match-id", default=None,
                    help="identificador del partido en la tabla de features "
                         "(override de --match; el split de build_dataset es por este id)")
    ap.add_argument("--labels", default=None,
                    help="CSV ground truth del label_tool (opcional; override de --match)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.match and not (args.tracking and args.match_id and args.out):
        ap.error("pasa --match <partido> o --tracking, --match-id y --out")

    match_id = args.match_id or args.match
    tracking = args.tracking or matchpaths.tracking_path(args.match)
    labels = args.labels or (matchpaths.labeled_path(args.match) if args.match else None)
    if labels and not os.path.exists(os.path.expanduser(labels)):
        labels = None
    out = args.out or matchpaths.features_path(args.match)

    rows = extract(tracking, match_id, labels)

    out = os.path.expanduser(out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    labelled = sum(1 for r in rows if r["label"] not in ("", "NONE"))
    print(f"{len(rows)} transiciones -> {out}"
          + (f" ({labelled} con evento, {len(rows) - labelled} NONE)"
             if labels else " (sin labels)"))


if __name__ == "__main__":
    main()
