"""Despega la pelota de los objetos INMOVILES a los que se engancha el detector.

EL PROBLEMA
El detector confunde la pelota con objetos blancos y quietos: la marca de penal
sobre todo, pero tambien el punto central, botellas y toallas del costado,
banderines y las pelotas de calentamiento. MEDIDO en spain-france (89.027
frames): el 23,5% de los frames con pelota estan CLAVADOS -- la posicion no se
mueve mas de 10 cm entre frames -- y el peor enganche dura 543 frames
SEGUIDOS, o sea 36 segundos con "la pelota" inmovil en el mismo punto.

POR QUE EL FILTRO EN LINEA NO ALCANZA
``_pick_ball`` es causal: acepta candidatos dentro de un radio medido desde la
ULTIMA posicion aceptada. Cuando esa ultima posicion es una marca pintada, el
enganche se auto-alimenta: la marca esta a distancia CERO y siempre entra en el
radio, mientras que la pelota real -- que se fue jugando a treinta metros --
queda fuera y se descarta como fisicamente imposible. Mirando solo hacia atras
no hay forma de saber cual de los dos es la pelota. Este script mira el partido
ENTERO, donde la diferencia es evidente.

LA SEÑAL
Una pelota real cruza una celda de 50 cm unas pocas veces en 90 minutos. Una
marca pintada aparece ahi CADA VEZ que entra en camara. Contar en cuantos
frames distintos aparece un candidato en cada celda separa las dos cosas sin
saber nada de geometria -- que es la clave, porque la version anterior ubicaba
la marca de penal por geometria (11 m del arco) y fallaba: la homografia tiene
varios metros de error y las rachas clavadas caian a 3,5-4,5 m de la posicion
nominal, tres veces el radio del veto.

POR QUE POR VENTANAS Y NO SOBRE TODO EL PARTIDO
El error de la homografia DERIVA a lo largo del partido, asi que un objeto fijo
se desparrama entre celdas y la señal se diluye. MEDIDO: en un clip de 3 minutos
la celda mas visitada concentra el 3,34% de los frames; sobre los 90 minutos
completos la mas visitada baja al 1,06%. En ventanas de 1500 frames (100 s a
15 fps) el error es localmente constante y la concentracion se recupera.

RESULTADO (spain-france completo, ventana 1500, umbral 0,8%)
    frames clavados   23,5%  ->   3,8%
    peor enganche      543   ->    13 frames (0,9 s)
    pelota presente   94,4%  ->  80,4%   (tras interpolar)
    movimientos imposibles 1,6% -> 1,9%  (sin cambio real)

Se pagan 14 puntos de presencia para sacar 20 puntos de posiciones FALSAS. Es
el mismo criterio que rige todo el pipeline: un frame en blanco lo puentea la
interpolacion; un frame con la pelota en el lugar equivocado inventa posesiones
y eventos que nunca pasaron.

NO HACE FALTA RE-TRACKEAR: trabaja sobre el sidecar de candidatos que main.py
ya escribio, asi que se puede correr sobre un tracking que ya existe.

Uso:
    python3 data_cleanup/ball_unpin.py \\
        --tracking-csv ~/football_data/matches/spain-france/tracking.csv \\
        --output       ~/football_data/matches/spain-france/tracking_unpin.csv
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

# Cancha reglamentaria; ver data_cleanup/pitch_config.py.
PITCH_L_CM, PITCH_W_CM = 10500.0, 6800.0

# Lado de la celda de conteo. A 50 cm una pelota en movimiento la cruza en uno
# o dos frames, asi que acumular decenas de frames en la misma celda solo le
# pasa a algo que no se mueve.
CELL_CM = 50.0
# Ventana de conteo, en frames. Ver la nota sobre la deriva de la homografia.
WINDOW_FRAMES = 1500
# Porcentaje de frames de la ventana que una celda tiene que acumular para
# quedar en la lista negra. Barrido sobre spain-france (frames clavados /
# presencia tras interpolar): 0,6% -> 1,8% / 77,8%; 0,8% -> 3,8% / 80,4%;
# 1,0% -> 4,9% / 81,8%; 1,5% -> 8,2% / 84,1%; 2,0% -> 11,4% / 86,3%. A 0,8% la
# curva de enganches ya esta aplanada y bajar mas solo cuesta cobertura.
BLACKLIST_PCT = 0.8
# Fisica del gate de continuidad (igual que main.py).
MAX_BALL_SPEED_MS = 40.0
# Margen fuera del cual un candidato se descarta. La pelota sale del campo de
# verdad (lateral, corner), asi que es generoso: corta botellas, no jugadas.
OFFPITCH_MARGIN = 0.04
# Umbral de "no se movio", para MEDIR (no para filtrar). Es el p10 del
# desplazamiento por frame medido sobre el clip: 3,4 cm. Un umbral mas grande
# no sirve -- 80 cm/frame a 15 fps son 12 m/s, o sea una pelota rodando normal.
STILL_CM = 10.0
STILL_MIN_RUN = 6


def read_fps(tracking, default=15.0):
    meta = tracking.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(meta):
        with open(meta) as fh:
            return float(json.load(fh).get("effective_fps") or default)
    return default


def offpitch(x, y, margin=OFFPITCH_MARGIN):
    return (x < -margin * PITCH_L_CM or x > (1 + margin) * PITCH_L_CM or
            y < -margin * PITCH_W_CM or y > (1 + margin) * PITCH_W_CM)


def cell(x, y):
    return int(x // CELL_CM), int(y // CELL_CM)


def build_blacklist(by_frame, n_frames, window, pct):
    """{indice_ventana: set(celdas)} con las celdas sobre-visitadas."""
    black = {}
    for w0 in range(1, n_frames + 1, window):
        cells, frames = defaultdict(set), set()
        for f in range(w0, min(w0 + window, n_frames + 1)):
            if f not in by_frame:
                continue
            frames.add(f)
            for _c, x, y, _b in by_frame[f]:
                cells[cell(x, y)].add(f)
        n = max(1, len(frames))
        black[w0 // window] = {k for k, v in cells.items()
                               if 100.0 * len(v) / n > pct}
    return black


def select_path(by_frame, n_frames, black, window, fps):
    """Re-elige la pelota frame a frame con la lista negra puesta."""
    last, gap, out = None, 1, {}
    for f in range(1, n_frames + 1):
        bad = black.get((f - 1) // window, set())
        radius = (MAX_BALL_SPEED_MS * max(1, gap) / fps) * 100.0
        best = None
        for conf, x, y, box in by_frame.get(f, ()):
            if offpitch(x, y):
                continue
            # Descarte DURO, no desempate: en el enganche el objeto quieto suele
            # ser el UNICO candidato del frame, asi que dejarlo competir es
            # dejarlo ganar.
            if cell(x, y) in bad:
                continue
            if last is None:
                d = 0.0
            else:
                d = math.hypot(x - last[0], y - last[1])
                if d > radius:
                    continue
            key = (conf, -d)
            if best is None or key > best[0]:
                best = (key, x, y, box)
        if best is None:
            gap += 1
            continue
        # El bbox viaja con la posicion: el label_tool dibuja el marcador desde
        # ahi, asi que dejar el viejo pondria el circulito sobre el objeto que
        # justamente acabamos de descartar.
        out[f] = (best[1], best[2], best[3])
        last, gap = (best[1], best[2]), 1
    return out


def pinned_stats(path):
    """(% de frames clavados, enganche mas largo) sobre un dict frame->(x,y)."""
    # Acepta valores (x, y) o (x, y, bbox): solo se miran las dos primeras.
    seen = sorted((f, (v[0], v[1])) for f, v in path.items())
    runs, start = [], 0
    for i in range(1, len(seen)):
        (f0, (x0, y0)), (f1, (x1, y1)) = seen[i - 1], seen[i]
        if not (f1 - f0 <= 2 and math.hypot(x1 - x0, y1 - y0) < STILL_CM):
            if i - start >= STILL_MIN_RUN:
                runs.append(i - start)
            start = i
    if len(seen) - start >= STILL_MIN_RUN:
        runs.append(len(seen) - start)
    return (100.0 * sum(runs) / max(1, len(seen)),
            max(runs) if runs else 0, len(runs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-csv", required=True)
    p.add_argument("--candidates", default=None,
                   help="sidecar de candidatos; por defecto "
                        "<tracking sin _*>_ball_candidates.csv")
    p.add_argument("--output", required=True)
    p.add_argument("--window", type=int, default=WINDOW_FRAMES)
    p.add_argument("--pct", type=float, default=BLACKLIST_PCT)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    tracking = os.path.expanduser(args.tracking_csv)
    cands = os.path.expanduser(args.candidates) if args.candidates else \
        tracking.rsplit(".", 1)[0] + "_ball_candidates.csv"
    if not os.path.exists(cands):
        raise SystemExit(
            f"No encuentro el sidecar de candidatos: {cands}\n"
            "Lo escribe main.py junto al tracking. Pasalo con --candidates.")

    fps = read_fps(tracking)
    by_frame = defaultdict(list)
    for r in csv.DictReader(open(cands)):
        by_frame[int(r["Frame"])].append(
            (float(r["Conf"]), float(r["X_Pitch"]), float(r["Y_Pitch"]),
             (r["X1"], r["Y1"], r["X2"], r["Y2"])))

    rows = list(csv.DictReader(open(tracking)))
    fields = rows[0].keys() if rows else []
    n_frames = max((int(r["Frame"]) for r in rows), default=0)

    antes = {}
    for r in rows:
        if r["Object"] == "ball":
            x, y = float(r["X_Pitch"]), float(r["Y_Pitch"])
            if not (x == 0 and y == 0):
                antes[int(r["Frame"])] = (x, y)

    black = build_blacklist(by_frame, n_frames, args.window, args.pct)
    n_cells = sum(len(v) for v in black.values())
    despues = select_path(by_frame, n_frames, black, args.window, fps)

    pa, ra, na = pinned_stats(antes)
    pd_, rd, nd = pinned_stats(despues)
    print(f"fps efectivo {fps:g} | ventana {args.window} frames "
          f"({args.window / fps:.0f} s) | umbral {args.pct}%")
    print(f"celdas en lista negra: {n_cells} "
          f"(en {len(black)} ventanas)\n")
    print(f"{'':<22}{'ANTES':>10}{'AHORA':>10}")
    print(f"{'pelota presente':<22}{100*len(antes)/max(1,n_frames):>9.1f}%"
          f"{100*len(despues)/max(1,n_frames):>9.1f}%")
    print(f"{'frames CLAVADOS':<22}{pa:>9.1f}%{pd_:>9.1f}%")
    print(f"{'peor enganche':<22}{ra:>9} {rd:>9}   frames "
          f"({ra/fps:.1f} s -> {rd/fps:.1f} s)")
    print(f"{'enganches':<22}{na:>9} {nd:>9}")

    if args.dry_run:
        print("\n--dry-run: no se escribio nada.")
        return

    # Reescribe SOLO las filas de pelota; los jugadores no se tocan.
    for r in rows:
        if r["Object"] != "ball":
            continue
        f = int(r["Frame"])
        if f in despues:
            x, y, box = despues[f]
            r["X_Pitch"], r["Y_Pitch"] = f"{x:.1f}", f"{y:.1f}"
            r["X1"], r["Y1"], r["X2"], r["Y2"] = box
        else:
            # Frame en blanco: es un estado valido del pipeline y la
            # interpolacion puentea los huecos cortos.
            for c in ("X1", "Y1", "X2", "Y2", "X_Pitch", "Y_Pitch"):
                r[c] = "0"

    out = os.path.expanduser(args.output)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {out}")

    src = tracking.rsplit(".", 1)[0] + ".meta.json"
    dst = out.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(src):
        with open(src) as a, open(dst, "w") as b:
            b.write(a.read())
        print(f"Sidecar: {dst}")


if __name__ == "__main__":
    main()
