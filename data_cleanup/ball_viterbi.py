"""Reconstruye la trayectoria de la pelota eligiendo el camino GLOBALMENTE
mas plausible entre todos los candidatos, con programacion dinamica (Viterbi).

POR QUE NO ALCANZA EL FILTRO EN LINEA
``main.py`` decide frame a frame mirando solo hacia atras (el gate de
continuidad de ``_pick_ball``). Eso tiene dos fallas estructurales:

  * si en un frame acepta un objeto equivocado (el punto de penal, una linea),
    a partir de ahi mide todo contra esa posicion falsa y RECHAZA la pelota
    real por "inalcanzable": el error se propaga;
  * cuando ningun candidato es alcanzable descarta el frame entero, aunque la
    pelota real estuviera entre los candidatos.

Un solver global no elige "el mejor candidato de este frame" sino "la mejor
SECUENCIA de candidatos de todo el partido". Si tomar la pelota real (conf
0.2) encaja con los frames vecinos y el marcador (conf 0.9) no lleva a ningun
lado, el camino coherente gana. Es asi como resuelven esto los sistemas
profesionales de tracking.

MODELO
Estados por frame: los candidatos guardados + un estado MISSING (la pelota no
se detecto). Costos (menor = mejor):

  * emision: ``-log(confianza)`` -- una deteccion floja cuesta, pero no manda;
  * transicion: velocidad implicita entre dos candidatos consecutivos.
    Por encima de ``--max-speed`` es INFINITO (fisicamente imposible); por
    debajo crece suave, para preferir trayectorias mansas frente a zigzags;
  * MISSING: un costo fijo por frame sin pelota, que evita que el camino
    "desaparezca" para esquivar cualquier decision dificil.

Uso:
    python3 data_cleanup/ball_viterbi.py \\
        --tracking-csv ~/football_data/matches/<m>/tracking.csv \\
        --candidates   ~/football_data/matches/<m>/tracking_ball_candidates.csv \\
        --output       ~/football_data/matches/<m>/tracking_viterbi.csv
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

PITCH_L_CM, PITCH_W_CM = 12000.0, 7000.0
PITCH_L_M, PITCH_W_M = 105.0, 68.0

MAX_SPEED_MS = 40.0      # por encima: transicion imposible
MISSING_COST = 2.5       # costo de declarar "sin pelota" en un frame
SPEED_WEIGHT = 0.05      # penalizacion por m/s POR ENCIMA de lo plausible
PLAUSIBLE_SPEED_MS = 20.0  # hasta aca la pelota se mueve normal: coste cero
MAX_MISSING_RUN = 90     # frames sin pelota tras los cuales se corta el enlace


def read_fps(tracking_csv, default=24.0):
    meta = tracking_csv.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(meta):
        with open(meta) as fh:
            return float(json.load(fh).get("effective_fps") or default)
    print(f"AVISO: no encuentro {meta}; asumo {default:g} fps.")
    return default


def load_candidates(path):
    """{frame: [(conf, x1, y1, x2, y2, x_pitch_m, y_pitch_m), ...]}"""
    by_frame = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                f = int(row["Frame"])
                by_frame[f].append((
                    float(row["Conf"]),
                    float(row["X1"]), float(row["Y1"]),
                    float(row["X2"]), float(row["Y2"]),
                    float(row["X_Pitch"]) / PITCH_L_CM * PITCH_L_M,
                    float(row["Y_Pitch"]) / PITCH_W_CM * PITCH_W_M,
                ))
            except (KeyError, TypeError, ValueError):
                continue
    return by_frame


def static_penalties(by_frame, cell_m=0.6, min_hits=None, weight=3.0):
    """Penaliza posiciones que REAPARECEN todo el partido en el mismo lugar.

    Es la senal que un filtro causal no puede usar: el punto de penal, una
    linea o un cartel aparecen como candidatos en el mismo punto durante todo
    el partido; la pelota real nunca vuelve al mismo centimetro cientos de
    veces. Sin esto el solver prefiere justamente el objeto quieto, porque
    quedarse inmovil no cuesta nada en la transicion y las marcas suelen tener
    MAS confianza que una pelota borrosa.

    Devuelve {(frame, idx_candidato): penalizacion}.
    """
    counts = defaultdict(int)
    for f, cands in by_frame.items():
        seen = set()
        for c in cands:
            key = (round(c[5] / cell_m), round(c[6] / cell_m))
            if key not in seen:      # una vez por frame, no infla por duplicados
                counts[key] += 1
                seen.add(key)

    n_frames = max(1, len(by_frame))
    if min_hits is None:
        # Una posicion presente en mas del 2% de los frames del partido no es
        # una pelota en juego: es parte de la cancha.
        min_hits = max(20, int(0.02 * n_frames))

    pen = {}
    for f, cands in by_frame.items():
        for i, c in enumerate(cands):
            key = (round(c[5] / cell_m), round(c[6] / cell_m))
            hits = counts[key]
            if hits > min_hits:
                pen[(f, i)] = weight * math.log(hits / min_hits + 1.0)
    return pen


def solve(by_frame, fps, max_speed=MAX_SPEED_MS):
    """Viterbi sobre los candidatos. Devuelve {frame: candidato elegido}."""
    frames = sorted(by_frame)
    if not frames:
        return {}

    # Estado: indice de candidato, o None para MISSING.
    # cost[j] = mejor costo acumulado terminando en el candidato j de este frame.
    pen = static_penalties(by_frame)

    prev_frame = None
    # Estado: (cand_idx | None, coste, backpointer, ancla)
    # El ancla es (frame, x, y) de la ULTIMA posicion real del camino. El estado
    # MISSING la arrastra para que, al volver a ver la pelota, se siga exigiendo
    # que el salto sea alcanzable desde donde estaba: sin esto el camino puede
    # "desaparecer" un momento y reaparecer en cualquier punto de la cancha
    # (medido: la cobertura subia a 89.2% pero los imposibles a 7.4%).
    prev_states = [(None, 0.0, None, None)]
    back = []

    for f in frames:
        cands = by_frame[f]
        gap = 1 if prev_frame is None else f - prev_frame
        # Radio fisico permitido para este salto temporal.
        reach = max_speed * gap / fps

        cur_states = []
        cur_back = []
        # --- candidatos reales ---
        for j, c in enumerate(cands):
            emis = -math.log(max(1e-3, min(1.0, c[0]))) + pen.get((f, j), 0.0)
            best_cost, best_ptr = float("inf"), None
            for k, (pidx, pcost, _p, anchor) in enumerate(prev_states):
                if pidx is None:
                    if anchor is None:
                        trans = 0.0          # todavia no vimos la pelota nunca
                    else:
                        af, ax, ay = anchor
                        elapsed = max(1, f - af)
                        d = math.hypot(c[5] - ax, c[6] - ay)
                        if d > max_speed * elapsed / fps:
                            continue         # inalcanzable desde la ultima real
                        v = d / (elapsed / fps)
                        trans = SPEED_WEIGHT * max(0.0, v - PLAUSIBLE_SPEED_MS)
                else:
                    pc = by_frame[prev_frame][pidx]
                    d = math.hypot(c[5] - pc[5], c[6] - pc[6])
                    if d > reach:
                        continue          # imposible: no se enlaza
                    # Solo se penaliza acercarse al limite fisico. Antes se
                    # penalizaba cualquier velocidad, lo que premiaba a los
                    # objetos INMOVILES -- exactamente los falsos positivos.
                    v = d / (gap / fps)
                    trans = SPEED_WEIGHT * max(0.0, v - PLAUSIBLE_SPEED_MS)
                total = pcost + trans
                if total < best_cost:
                    best_cost, best_ptr = total, k
            if best_ptr is not None:
                cur_states.append((j, best_cost + emis, best_ptr, (f, c[5], c[6])))
                cur_back.append(best_ptr)

        # --- estado MISSING ---
        best_cost, best_ptr, best_anchor = float("inf"), None, None
        for k, (pidx, pcost, _p, anchor) in enumerate(prev_states):
            if pcost < best_cost:
                best_cost, best_ptr = pcost, k
                # Al entrar en MISSING se hereda el ancla; si venimos de un
                # candidato real, ese candidato PASA a ser el ancla.
                best_anchor = ((prev_frame, by_frame[prev_frame][pidx][5],
                                by_frame[prev_frame][pidx][6])
                               if pidx is not None else anchor)
        cur_states.append((None, best_cost + MISSING_COST, best_ptr, best_anchor))
        cur_back.append(best_ptr)

        back.append((f, list(cur_states)))
        prev_states = cur_states
        prev_frame = f

    # Backtracking desde el estado final mas barato.
    path = {}
    idx = min(range(len(prev_states)), key=lambda i: prev_states[i][1])
    for f, states in reversed(back):
        cand_idx, _cost, ptr, _anchor = states[idx]
        if cand_idx is not None:
            path[f] = by_frame[f][cand_idx]
        idx = ptr if ptr is not None else 0
    return path


def impossible_fraction(path, fps, limit=35.0):
    fs = sorted(path)
    bad = tot = 0
    for a, b in zip(fs, fs[1:]):
        if not 1 <= b - a <= 3:
            continue
        tot += 1
        d = math.hypot(path[b][5] - path[a][5], path[b][6] - path[a][6])
        if d / ((b - a) / fps) > limit:
            bad += 1
    return 100.0 * bad / max(1, tot)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-csv", required=True)
    p.add_argument("--candidates", default=None,
                   help="default: <tracking>_ball_candidates.csv")
    p.add_argument("--output", required=True)
    p.add_argument("--max-speed", type=float, default=MAX_SPEED_MS)
    args = p.parse_args()

    tracking = os.path.expanduser(args.tracking_csv)
    cand_path = os.path.expanduser(
        args.candidates or tracking.rsplit(".", 1)[0] + "_ball_candidates.csv")
    if not os.path.exists(cand_path):
        sys.exit(f"No encuentro los candidatos: {cand_path}\n"
                 f"Se generan al trackear con la version que escribe "
                 f"'<video>_ball_candidates.csv'. Hay que re-trackear.")

    fps = read_fps(tracking)
    by_frame = load_candidates(cand_path)
    print(f"{sum(len(v) for v in by_frame.values())} candidatos en "
          f"{len(by_frame)} frames, fps={fps:g}")

    path = solve(by_frame, fps, args.max_speed)
    print(f"Camino resuelto: pelota en {len(path)} frames "
          f"({100.0 * len(path) / max(1, len(by_frame)):.1f}% de los frames con "
          f"candidatos)")
    print(f"Movimientos imposibles en el camino: "
          f"{impossible_fraction(path, fps):.1f}%")

    rows, n_ball, changed = [], 0, 0
    with open(tracking) as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        for row in reader:
            if row["Object"] == "ball":
                n_ball += 1
                f = int(row["Frame"])
                chosen = path.get(f)
                before = row["X_Pitch"]
                if chosen is None:
                    for c in ("X1", "Y1", "X2", "Y2", "X_Pitch", "Y_Pitch"):
                        row[c] = "0"
                else:
                    row["X1"], row["Y1"] = f"{chosen[1]:.1f}", f"{chosen[2]:.1f}"
                    row["X2"], row["Y2"] = f"{chosen[3]:.1f}", f"{chosen[4]:.1f}"
                    row["X_Pitch"] = f"{chosen[5] / PITCH_L_M * PITCH_L_CM:.1f}"
                    row["Y_Pitch"] = f"{chosen[6] / PITCH_W_M * PITCH_W_CM:.1f}"
                if row["X_Pitch"] != before:
                    changed += 1
            rows.append(row)

    out = os.path.expanduser(args.output)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nFilas de pelota reescritas: {changed} de {n_ball}")
    print(f"CSV: {out}")

    src = tracking.rsplit(".", 1)[0] + ".meta.json"
    dst = out.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(src) and not os.path.exists(dst):
        with open(src) as a, open(dst, "w") as b:
            b.write(a.read())
        print(f"Sidecar: {dst}")


if __name__ == "__main__":
    main()
