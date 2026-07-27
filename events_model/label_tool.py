"""
Herramienta de etiquetado por correccion de reglas.

Muestra cada evento propuesto por las reglas junto al video (con las cajas del
tracking superpuestas) y deja aceptar / rechazar / reetiquetar / agregar eventos
con el teclado. Exporta un ground truth en el mismo formato Metrica del pipeline
mas columnas de auditoria (Verdict, Original Type).

Uso (convencion por partido, resuelve las rutas solo):
    python3 events_model/label_tool.py --match psg_bayern

Uso (rutas explicitas):
    python3 events_model/label_tool.py \
        --video  ~/football_data/matches/psg_bayern/video.mp4 \
        --events events_model/dataset/psg_bayern_proposed.csv \
        --tracking ~/football_data/matches/psg_bayern/tracking.csv

El progreso se guarda solo (.review.json) despues de cada decision: se puede
cortar y retomar cuando sea. Al salir (q) exporta el ground truth a
<match>_labeled.csv (o <events>_groundtruth.csv en modo de rutas explicitas).

Teclas (pensadas para etiquetar rapido, mano derecha en el teclado):
    ESPACIO     play / pausa
    , / .       un frame atras / adelante (pausado)
    h / l       saltar -1s / +1s
    p / n       evento anterior / siguiente (n salta al proximo pendiente)
    y           ACEPTAR el evento tal cual
    x           RECHAZAR (queda como negativo para entrenar)
    r           REETIQUETAR -> apretar la clase (1-8)
    a           AGREGAR evento en el frame actual -> clase (1-8) -> equipo (1/2)
    u           deshacer la ultima decision
    b           mostrar/ocultar cajas del tracking
    [ / ]       mas lento / mas rapido
    s           guardar ahora
    q           guardar, exportar ground truth y salir
    ESC         cancelar reetiquetado/agregado en curso

Clases:  1=PASS  2=BALL LOST  3=RECOVERY  4=SHOT  5=GOAL  6=FOUL
         7=SET PIECE  8=DUEL      Equipos: 1=Home  2=Away
"""

import argparse
import csv
import json
import os
import sys

import cv2

import matchpaths

CLASS_KEYS = {
    ord("1"): "PASS",
    ord("2"): "BALL LOST",
    ord("3"): "RECOVERY",
    ord("4"): "SHOT",
    ord("5"): "GOAL",
    ord("6"): "FOUL",
    ord("7"): "SET PIECE",
    ord("8"): "DUEL",
}
TEAM_KEYS = {ord("1"): "Home", ord("2"): "Away"}

# Segundos de contexto que se muestran antes/despues del evento.
PRE_ROLL_S = 2.0
POST_ROLL_S = 2.0

HEADER = [
    "Team", "Type", "Subtype", "Period",
    "Start Frame", "Start Time [s]", "End Frame", "End Time [s]",
    "From", "To", "Start X", "Start Y", "End X", "End Y",
]

TEAM_COLORS = {"0": (60, 180, 255), "1": (255, 120, 60)}  # BGR por team raw


def load_events(path):
    """Lee el CSV de eventos (formato Metrica) a una lista de dicts."""
    events = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append({
                "team": row.get("Team", ""),
                "type": row.get("Type", ""),
                "subtype": row.get("Subtype", ""),
                "period": row.get("Period", "1"),
                "start_frame": int(float(row["Start Frame"])),
                "start_time": row.get("Start Time [s]", ""),
                "end_frame": int(float(row["End Frame"])),
                "end_time": row.get("End Time [s]", ""),
                "from": row.get("From", ""),
                "to": row.get("To", ""),
                "start_x": row.get("Start X", ""),
                "start_y": row.get("Start Y", ""),
                "end_x": row.get("End X", ""),
                "end_y": row.get("End Y", ""),
            })
    return events


def load_tracking(path):
    """Indexa el CSV crudo de tracking por frame: {frame: [(obj, id, team, bbox)]}.

    Formato: Frame, Object, Object ID, Team, X1, Y1, X2, Y2, X_Pitch, Y_Pitch
    (se lee por posicion: el header exportado tiene un typo, dos columnas "X2").
    """
    by_frame = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 8:
                continue
            try:
                frame = int(float(row[0]))
                bbox = tuple(int(float(v)) for v in row[4:8])
            except ValueError:
                continue
            if bbox == (0, 0, 0, 0):
                continue
            by_frame.setdefault(frame, []).append(
                (row[1], row[2], str(row[3]).split(".")[0], bbox))
    return by_frame


def find_suspicious_ball_frames(by_frame, fps, still_px=2.0, min_still_s=1.5,
                                max_speed_px=None):
    """Frames donde NO hay que fiarse del marcador de la pelota.

    Dos firmas, ambas imposibles para una pelota en juego:

    * INMOVIL: clavada al pixel durante segundos -> es una marca de la cancha
      (el punto de penal es el caso tipico).
    * TELETRANSPORTE: salta hacia/desde el frame vecino a una velocidad
      imposible -> ese frame el detector se engancho a otra cosa.

    NO se borra la deteccion (borrarlas empeora los eventos: las rachas mas
    largas eran la pelota real durante pausas). Solo se dibujan distinto para
    que no te guien mal al etiquetar.
    """
    balls = []
    for frame in sorted(by_frame):
        for obj, _oid, _team, (x1, y1, x2, y2) in by_frame[frame]:
            if obj == "ball":
                balls.append((frame, (x1 + x2) / 2, (y1 + y2) / 2))
                break

    suspicious = set()

    # --- inmovil ---
    run = []
    min_len = max(2, int(min_still_s * fps))

    def flush():
        if len(run) >= min_len:
            suspicious.update(f for f, _, _ in run)

    for f, cx, cy in balls:
        if not run:
            run.append((f, cx, cy))
            continue
        pf, px, py = run[-1]
        if f - pf <= 2 and abs(cx - px) < still_px and abs(cy - py) < still_px:
            run.append((f, cx, cy))
        else:
            flush()
            run = [(f, cx, cy)]
    flush()

    # --- teletransporte ---
    # Sin escala de cancha usamos un umbral en pixeles: un tiro potente recorre
    # ~35 m/s, que sobre un plano abierto son del orden de 600 px/s.
    if max_speed_px is None:
        max_speed_px = 600.0
    for i in range(1, len(balls)):
        (f0, x0, y0), (f1, x1, y1) = balls[i - 1], balls[i]
        df = f1 - f0
        if not 1 <= df <= 3:
            continue
        v = (((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5) / (df / fps)
        if v > max_speed_px:
            # No se sabe cual de los dos es el impostor: se marcan ambos.
            suspicious.add(f0)
            suspicious.add(f1)
    return suspicious


class ReviewState:
    """Decisiones tomadas + eventos agregados; persiste a JSON tras cada cambio."""

    def __init__(self, path):
        self.path = path
        self.decisions = {}   # idx (str) -> {"verdict": ..., "new_type": ...}
        self.added = []       # eventos creados a mano
        self.history = []     # para undo: ("decision", idx) | ("added", n)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            self.decisions = data.get("decisions", {})
            self.added = data.get("added", [])

    def save(self):
        with open(self.path, "w") as f:
            json.dump({"decisions": self.decisions, "added": self.added}, f,
                      indent=1)

    def decide(self, idx, verdict, new_type=None):
        self.decisions[str(idx)] = {"verdict": verdict, "new_type": new_type}
        self.history.append(("decision", str(idx)))
        self.save()

    def add_event(self, event):
        self.added.append(event)
        self.history.append(("added", len(self.added) - 1))
        self.save()

    def undo(self):
        if not self.history:
            return None
        kind, key = self.history.pop()
        if kind == "decision":
            self.decisions.pop(key, None)
        else:
            self.added.pop(key)
        self.save()
        return kind

    def verdict(self, idx):
        d = self.decisions.get(str(idx))
        return d["verdict"] if d else None


class LabelTool:
    def __init__(self, video_path, events, tracking, state, fps, frame_stride=1):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            sys.exit(f"No pude abrir el video: {video_path}")
        # ``self.vframe`` cuenta frames del CSV/tracking, NO del video: los
        # eventos y las cajas estan indexados asi. Con --frame-stride los dos
        # numeros dejan de coincidir (el frame 1000 del CSV es el 1999 del
        # video), y usar uno por el otro desincroniza el video respecto de las
        # cajas -- la pelota aparece en cualquier lado menos donde esta.
        self.frame_stride = max(1, int(frame_stride))
        self.fps = fps or ((self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
                           / self.frame_stride)
        self.total = (int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                      // self.frame_stride)
        self.events = events
        self.tracking = tracking
        self.static_ball = find_suspicious_ball_frames(tracking, self.fps)
        self.state = state
        self.idx = self._first_pending()
        self.vframe = 0          # frame de video actual (0-based)
        self.playing = True
        self.speed = 1.0
        self.show_boxes = True
        self.mode = "review"     # review | relabel | add_class | add_team
        self.pending_add = None
        self.raw = None          # ultimo frame leido (sin HUD)
        self.msg = ""

    # ---------------- navegacion ----------------
    def _first_pending(self):
        for i in range(len(self.events)):
            if self.state.verdict(i) is None:
                return i
        return 0

    def window(self):
        ev = self.events[self.idx]
        pre = int(PRE_ROLL_S * self.fps)
        post = int(POST_ROLL_S * self.fps)
        start = max(0, ev["start_frame"] - 1 - pre)
        end = min(self.total - 1, ev["end_frame"] - 1 + post)
        return start, end

    def goto_event(self, idx):
        self.idx = max(0, min(len(self.events) - 1, idx))
        self.seek(self.window()[0])
        self.playing = True

    def next_pending(self):
        n = len(self.events)
        for step in range(1, n + 1):
            i = (self.idx + step) % n
            if self.state.verdict(i) is None:
                self.goto_event(i)
                return
        self.msg = "No quedan eventos pendientes — q para exportar"

    def seek(self, vframe):
        self.vframe = max(0, min(self.total - 1, vframe))
        # Unico lugar donde se traduce a coordenadas de video.
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.vframe * self.frame_stride)
        ok, frame = self.cap.read()
        if ok:
            self.raw = frame

    def advance(self):
        """Avanzar UN frame del CSV = ``frame_stride`` frames del video."""
        self.vframe += 1
        ok, frame = None, None
        for _ in range(self.frame_stride):
            ok, frame = self.cap.read()
            if not ok:
                return False
        self.raw = frame
        return True

    def step(self, delta):
        self.playing = False
        self.seek(self.vframe + delta)

    # ---------------- decisiones ----------------
    def decide(self, verdict, new_type=None):
        self.state.decide(self.idx, verdict, new_type)
        self.msg = f"#{self.idx}: {verdict}" + (f" -> {new_type}" if new_type else "")
        self.next_pending()

    def add(self, cls, team):
        t = round((self.vframe + 1) / self.fps, 3)
        self.state.add_event({
            "team": team, "type": cls, "subtype": "MANUAL",
            "period": "1",
            "start_frame": self.vframe + 1, "start_time": t,
            "end_frame": self.vframe + 1, "end_time": t,
            "from": "", "to": "", "start_x": "", "start_y": "",
            "end_x": "", "end_y": "",
        })
        self.msg = f"AGREGADO {cls} ({team}) en frame {self.vframe + 1}"

    # ---------------- dibujo ----------------
    def draw(self):
        if self.raw is None:
            return None
        frame = self.raw.copy()
        ev = self.events[self.idx]

        if self.show_boxes:
            highlight = {ev["from"].replace("Player ", ""): (0, 255, 0),
                         ev["to"].replace("Player ", ""): (255, 255, 0)}
            for obj, oid, team, (x1, y1, x2, y2) in \
                    self.tracking.get(self.vframe + 1, []):
                if obj == "ball":
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if (self.vframe + 1) in self.static_ball:
                        # Inmovil = casi seguro el punto de penal u otra marca
                        # fija. Se dibuja apagado y con "?" para que no te guie:
                        # la pelota real esta en otro lado, buscala con el ojo.
                        cv2.circle(frame, (cx, cy), 10, (120, 120, 120), 1)
                        cv2.putText(frame, "?", (cx + 12, cy + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
                    else:
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 255), 2)
                    continue
                color = highlight.get(oid) or TEAM_COLORS.get(team, (200, 200, 200))
                thick = 2 if oid in highlight else 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                cv2.putText(frame, oid, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # HUD
        h = frame.shape[0]
        done = len(self.state.decisions)
        verdict = self.state.verdict(self.idx) or "PENDIENTE"
        lines = [
            f"[{self.idx + 1}/{len(self.events)}] {ev['team']} {ev['type']}"
            f"{('/' + ev['subtype']) if ev['subtype'] else ''}"
            f"  {ev['from']}{(' -> ' + ev['to']) if ev['to'] else ''}"
            f"  frames {ev['start_frame']}-{ev['end_frame']}   [{verdict}]",
            f"frame {self.vframe + 1}  t={(self.vframe + 1) / self.fps:6.1f}s"
            f"  x{self.speed:.2g}  revisados {done}/{len(self.events)}"
            f"  agregados {len(self.state.added)}",
        ]
        if self.mode == "relabel":
            lines.append("REETIQUETAR: 1 PASS 2 BALL-LOST 3 RECOVERY 4 SHOT"
                         " 5 GOAL 6 FOUL 7 SET-PIECE 8 DUEL  (ESC cancela)")
        elif self.mode == "add_class":
            lines.append("AGREGAR: clase? 1 PASS 2 BALL-LOST 3 RECOVERY 4 SHOT"
                         " 5 GOAL 6 FOUL 7 SET-PIECE 8 DUEL  (ESC cancela)")
        elif self.mode == "add_team":
            lines.append(f"AGREGAR {self.pending_add}: equipo? 1 Home  2 Away"
                         "  (ESC cancela)")
        else:
            lines.append("y acepta  x rechaza  r reetiqueta  a agrega  u undo"
                         "  n/p evento  espacio pausa  ,/. frame  h/l 1s  q sale")
        if self.msg:
            lines.append(self.msg)

        y = h - 14 * len(lines) - 10
        cv2.rectangle(frame, (0, y - 18), (frame.shape[1], h), (0, 0, 0), -1)
        for line in lines:
            cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
            y += 14
        return frame

    # ---------------- loop principal ----------------
    def run(self):
        cv2.namedWindow("label", cv2.WINDOW_NORMAL)
        self.goto_event(self.idx)
        while True:
            start, end = self.window()
            if self.playing:
                if self.vframe >= end:
                    self.seek(start)  # loop del clip del evento
                elif not self.advance():
                    self.seek(start)

            display = self.draw()
            if display is not None:
                cv2.imshow("label", display)

            delay = max(1, int(1000 / (self.fps * self.speed))) if self.playing else 30
            key = cv2.waitKey(delay) & 0xFF
            if key == 255:
                continue
            self.msg = ""

            if self.mode == "relabel":
                if key in CLASS_KEYS:
                    self.decide("RELABELED", CLASS_KEYS[key])
                    self.mode = "review"
                elif key == 27:
                    self.mode = "review"
                continue
            if self.mode == "add_class":
                if key in CLASS_KEYS:
                    self.pending_add = CLASS_KEYS[key]
                    self.mode = "add_team"
                elif key == 27:
                    self.mode = "review"
                continue
            if self.mode == "add_team":
                if key in TEAM_KEYS:
                    self.add(self.pending_add, TEAM_KEYS[key])
                    self.mode = "review"
                elif key == 27:
                    self.mode = "review"
                continue

            if key == ord(" "):
                self.playing = not self.playing
            elif key == ord("."):
                self.step(1)
            elif key == ord(","):
                self.step(-1)
            elif key == ord("l"):
                self.seek(self.vframe + int(self.fps))
            elif key == ord("h"):
                self.seek(self.vframe - int(self.fps))
            elif key == ord("n"):
                self.next_pending()
            elif key == ord("p"):
                self.goto_event(self.idx - 1)
            elif key == ord("y"):
                self.decide("ACCEPTED")
            elif key == ord("x"):
                self.decide("REJECTED")
            elif key == ord("r"):
                self.mode = "relabel"
                self.playing = False
            elif key == ord("a"):
                self.mode = "add_class"
                self.playing = False
            elif key == ord("u"):
                kind = self.state.undo()
                self.msg = f"undo: {kind}" if kind else "nada para deshacer"
            elif key == ord("b"):
                self.show_boxes = not self.show_boxes
            elif key == ord("]"):
                self.speed = min(4.0, self.speed * 1.5)
            elif key == ord("["):
                self.speed = max(0.25, self.speed / 1.5)
            elif key == ord("s"):
                self.state.save()
                self.msg = "guardado"
            elif key == ord("q"):
                self.state.save()
                break
        cv2.destroyAllWindows()


def export_ground_truth(events, state, out_path):
    """Ground truth = propuestas revisadas + agregados, formato Metrica + auditoria.

    Los RECHAZADOS quedan en el CSV (Verdict=REJECTED): son negativos valiosos
    para entrenar. El codigo de entrenamiento filtra por Verdict.
    """
    rows = [HEADER + ["Verdict", "Original Type"]]
    merged = []
    for i, ev in enumerate(events):
        d = state.decisions.get(str(i))
        if d is None:
            continue  # sin revisar: no entra al ground truth
        etype = d["new_type"] or ev["type"]
        subtype = ev["subtype"] if d["verdict"] == "ACCEPTED" else ""
        merged.append((ev["start_frame"], [
            ev["team"], etype, subtype, ev["period"],
            ev["start_frame"], ev["start_time"], ev["end_frame"], ev["end_time"],
            ev["from"], ev["to"], ev["start_x"], ev["start_y"],
            ev["end_x"], ev["end_y"], d["verdict"], ev["type"],
        ]))
    for ev in state.added:
        merged.append((ev["start_frame"], [
            ev["team"], ev["type"], ev["subtype"], ev["period"],
            ev["start_frame"], ev["start_time"], ev["end_frame"], ev["end_time"],
            ev["from"], ev["to"], ev["start_x"], ev["start_y"],
            ev["end_x"], ev["end_y"], "ADDED", "",
        ]))
    merged.sort(key=lambda r: r[0])
    rows += [r for _, r in merged]
    with open(out_path, "w", newline="\n") as f:
        csv.writer(f).writerows(rows)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--match", default=None,
                    help="nombre del partido (convencion "
                         "~/football_data/matches/<match>/ + events_model/dataset/)")
    ap.add_argument("--video", default=None, help="override de --match")
    ap.add_argument("--events", default=None,
                    help="CSV de eventos propuesto por las reglas (override de --match)")
    ap.add_argument("--tracking", default=None,
                    help="CSV crudo de tracking para superponer las cajas (override de --match)")
    ap.add_argument("--fps", type=float, default=None,
                    help="override del fps EFECTIVO del tracking, o sea "
                         "fps_video/frame_stride (default: se deduce del video "
                         "y del sidecar)")
    args = ap.parse_args()

    if not args.match and not (args.video and args.events):
        ap.error("pasa --match <partido> o --video y --events")

    video = os.path.expanduser(args.video or matchpaths.video_path(args.match))
    events_path = os.path.expanduser(args.events or matchpaths.proposed_path(args.match))
    tracking_path = args.tracking or (matchpaths.tracking_path(args.match) if args.match else None)

    if args.match:
        review_path = matchpaths.review_path(args.match)
        labeled_out = matchpaths.labeled_path(args.match)
    else:
        base = events_path.rsplit(".", 1)[0]
        review_path = base + ".review.json"
        labeled_out = base + "_groundtruth.csv"

    events = load_events(events_path)
    if not events:
        sys.exit("El CSV de eventos esta vacio.")
    tracking = load_tracking(os.path.expanduser(tracking_path)) if tracking_path else {}

    os.makedirs(os.path.dirname(review_path) or ".", exist_ok=True)
    state = ReviewState(review_path)
    if state.decisions:
        print(f"Retomando: {len(state.decisions)} decisiones previas")

    # El stride decide que fotograma del video corresponde a cada frame del
    # CSV. Sin esto el video y las cajas van desfasados un factor del stride.
    frame_stride = 1
    if tracking_path:
        meta_path = os.path.expanduser(tracking_path).rsplit(".", 1)[0] + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                meta = json.load(fh)
            frame_stride = int(meta.get("frame_stride") or 1)
            print(f"Sidecar: frame_stride={frame_stride}, "
                  f"effective_fps={meta.get('effective_fps')}")
        else:
            print(f"AVISO: no encuentro {meta_path}; asumo frame_stride=1. "
                  f"Si el tracking uso --frame-stride, el video va a ir "
                  f"desincronizado de las cajas.")

    # Diagnostico de la fuente: si el marcador de la pelota miente, etiquetar es
    # imposible, y la causa mas comun no es el algoritmo sino estar leyendo un
    # tracking VIEJO. Se imprime que archivo se usa y que tan fiable es.
    if tracking:
        fps_eff = (args.fps or 0) or 15.0
        susp = find_suspicious_ball_frames(tracking, fps_eff)
        with_ball = sum(1 for f in tracking
                        if any(o == "ball" for o, *_ in tracking[f]))
        ev_ok = sum(1 for e in events
                    if e["start_frame"] in tracking
                    and e["start_frame"] not in susp
                    and any(o == "ball" for o, *_ in tracking[e["start_frame"]]))
        print(f"\nTracking: {tracking_path}")
        print(f"  frames con pelota: {100.0 * with_ball / max(1, len(tracking)):.1f}%"
              f" | sospechosos: {100.0 * len(susp) / max(1, with_ball):.1f}%")
        print(f"  eventos con pelota CONFIABLE: "
              f"{100.0 * ev_ok / max(1, len(events)):.1f}%")
        if 100.0 * ev_ok / max(1, len(events)) < 60:
            print("  ⚠️  Bajo. Si ya re-trackeaste con el fix de continuidad, "
                  "revisa que este CSV sea el NUEVO (¿lo copiaste sobre "
                  "tracking.csv?).")
        print()

    tool = LabelTool(video, events, tracking, state, args.fps, frame_stride)
    tool.run()

    out = export_ground_truth(events, state, labeled_out)
    total = len(state.decisions)
    print(f"Revisados {total}/{len(events)} + {len(state.added)} agregados")
    print(f"Ground truth -> {out}")


if __name__ == "__main__":
    main()
