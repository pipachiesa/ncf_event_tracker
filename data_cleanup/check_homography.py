"""Mide si el sistema de coordenadas de cancha es ESTABLE, usando los jugadores.

LA IDEA
Los veintipico de jugadores no se mueven todos juntos ni en la misma direccion.
Asi que el desplazamiento MEDIANO de todo el plantel entre dos frames
consecutivos deberia ser chico y parecido al movimiento real de una persona
(20-40 cm por frame a 15 fps). Cuando ese valor salta a varios metros, no se
movio el plantel: se movio la CANCHA debajo de ellos. Es una medida directa de
la calidad de la homografia que no necesita ground truth ni el video.

POR QUE IMPORTA MAS QUE CUALQUIER METRICA DE PELOTA
Todas las features del clasificador (distancias, velocidades, zonas) se calculan
en coordenadas de cancha. Si esas coordenadas saltan metros, ninguna limpieza
posterior de la pelota puede recuperar la informacion.

MEDIDO en spain-france (89.027 frames, homography_every=5): 6,36% de los frames
con salto global > 1 m y 1,67% con salto > 10 m, y el **93,6% de esos saltos cae
en frames multiplo de 5**, o sea exactamente cuando main.py re-estimaba la
transformacion. Con 4 keypoints (el minimo exacto) el error de reproyeccion es
cero por construccion, asi que era imposible detectar una mala solucion.

Uso:
    python3 data_cleanup/check_homography.py --tracking-csv <a.csv> [<b.csv> ...]
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

# Umbral de "esto no lo hizo un jugador". Un futbolista corriendo a 9 m/s a
# 15 fps recorre 60 cm por frame; un metro entero de TODO el plantel a la vez no
# tiene explicacion futbolistica.
JUMP_CM = 100.0
MIN_COMMON = 5      # ids compartidos minimos para que la mediana signifique algo


def read_fps(tracking, default=15.0):
    meta = tracking.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(meta):
        with open(meta) as fh:
            return float(json.load(fh).get("effective_fps") or default)
    return default


PITCH_L_CM, PITCH_W_CM = 12000.0, 7000.0
# Area de penal segun SoccerPitchConfiguration (la misma que usa main.py).
PBOX_L_CM, PBOX_W_CM = 2015.0, 4100.0
PBOX_Y0 = (PITCH_W_CM - PBOX_W_CM) / 2.0     # 1450
PBOX_Y1 = (PITCH_W_CM + PBOX_W_CM) / 2.0     # 5550
# Fraccion de la superficie que ocupan las DOS areas: el valor esperado si la
# pelota se repartiera por la cancha. Sale 19,6%.
PBOX_AREA_FRAC = 2 * PBOX_L_CM * PBOX_W_CM / (PITCH_L_CM * PITCH_W_CM)


def ball_in_penalty_box(path):
    """LA METRICA DEL SINTOMA que reporto Felipe.

    Con la homografia mal calibrada, los falsos positivos de pelota (la marca
    de penal, puntos blancos de afuera) se amontonan en las areas: medido
    58,6% de los frames con pelota, contra 19,6% esperado por superficie, o sea
    3x sobre-representada.

    Ojo con las metricas de radio alrededor de la MARCA de penal: dan 0,00% de
    forma sistematica porque los enganches caen a 6-14 m de la marca nominal.
    Hay que medir el area entera, no un circulito.
    """
    total = inside = 0
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["Object"] != "ball":
                continue
            try:
                x, y = float(r["X_Pitch"]), float(r["Y_Pitch"])
            except (TypeError, ValueError):
                continue
            if x == 0 and y == 0:
                continue
            total += 1
            if PBOX_Y0 <= y <= PBOX_Y1 and (x <= PBOX_L_CM
                                            or x >= PITCH_L_CM - PBOX_L_CM):
                inside += 1
    return total, inside


def calibration(path):
    """Chequeo ABSOLUTO: ¿el mapa apunta a la cancha correcta?

    La estabilidad no alcanza. Un mapa CONGELADO no salta nunca y saca nota
    perfecta en el test de estabilidad mientras pone al arquero a 39 m de su
    arco. Esto mira si las posiciones tienen sentido en terminos absolutos:
    sobre unos minutos de juego los jugadores tienen que repartirse por casi
    toda la cancha y casi nunca salirse de los limites.
    """
    xs, ys, off = [], [], 0
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["Object"] == "ball":
                continue
            try:
                x, y = float(r["X_Pitch"]), float(r["Y_Pitch"])
            except (TypeError, ValueError):
                continue
            if x == 0 and y == 0:
                continue
            xs.append(x)
            ys.append(y)
            if not (0 <= x <= PITCH_L_CM and 0 <= y <= PITCH_W_CM):
                off += 1
    return xs, ys, off


def analyse(path):
    frames = defaultdict(dict)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["Object"] == "ball":
                continue
            try:
                x, y = float(r["X_Pitch"]), float(r["Y_Pitch"])
            except (TypeError, ValueError):
                continue
            if x == 0 and y == 0:
                continue
            frames[int(r["Frame"])][r["Object ID"]] = (x, y)

    fs = sorted(frames)
    shifts = []
    for a, b in zip(fs, fs[1:]):
        if b - a != 1:
            continue
        common = frames[a].keys() & frames[b].keys()
        if len(common) < MIN_COMMON:
            continue
        d = sorted(math.hypot(frames[b][k][0] - frames[a][k][0],
                              frames[b][k][1] - frames[a][k][1])
                   for k in common)
        shifts.append((b, d[len(d) // 2]))
    return shifts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-csv", nargs="+", required=True)
    p.add_argument("--every", type=int, default=5,
                   help="homography_every con el que se trackeo (para el test "
                        "de periodicidad)")
    args = p.parse_args()

    for path in args.tracking_csv:
        path = os.path.expanduser(path)
        fps = read_fps(path)
        shifts = analyse(path)
        if not shifts:
            print(f"\n=== {os.path.basename(path)} ===\n  sin datos suficientes")
            continue
        vals = sorted(s for _f, s in shifts)
        n = len(vals)

        def q(x):
            return vals[int(x * (n - 1))]

        big = [f for f, s in shifts if s > JUMP_CM]
        print(f"\n=== {os.path.basename(path)}  (fps={fps:g}, {n} frames) ===")
        print("  desplazamiento mediano del plantel entre frames (cm)")
        print(f"    p50 {q(.50):8.1f}   p90 {q(.90):8.1f}   "
              f"p99 {q(.99):8.1f}    <- p50 sano ~15-40 cm")
        for t in (100, 200, 500, 1000):
            c = sum(1 for v in vals if v > t)
            print(f"    frames con salto > {t/100:>4.0f} m: {c:>6}  "
                  f"({100.0*c/n:5.2f}%)")

        # Si los saltos se concentran en los frames de refresco, la causa es la
        # re-estimacion de la homografia y no el tracking de jugadores.
        if big and args.every > 1:
            hits = sum(1 for f in big if f % args.every == 1)
            frac = 100.0 * hits / len(big)
            print(f"    de esos saltos, {frac:.1f}% caen en frames de refresco "
                  f"(cada {args.every}); al azar seria {100.0/args.every:.1f}%")
            if frac > 2 * 100.0 / args.every:
                print("    -> LA CAUSA ES LA HOMOGRAFIA, no el tracking.")

        # --- CALIBRACION (independiente de la estabilidad) ---
        xs, ys, off = calibration(path)
        if xs:
            xs.sort()
            ys.sort()
            m = len(xs)

            def qq(v, p):
                return v[int(p * (len(v) - 1))] / 100.0

            print("  CALIBRACION (que el mapa apunte a la cancha correcta)")
            print(f"    x: p01 {qq(xs,.01):6.1f}  p50 {qq(xs,.50):6.1f}  "
                  f"p99 {qq(xs,.99):6.1f} m   (cancha 0-120)")
            print(f"    y: p01 {qq(ys,.01):6.1f}  p50 {qq(ys,.50):6.1f}  "
                  f"p99 {qq(ys,.99):6.1f} m   (cancha 0-70)")
            print(f"    detecciones fuera de los limites: {100.0*off/m:.1f}%")
            malo = []
            if qq(xs, .01) > 10:
                malo.append(f"nadie aparece nunca en los primeros "
                            f"{qq(xs,.01):.0f} m")
            if qq(xs, .99) < 110:
                malo.append(f"nadie pasa nunca de {qq(xs,.99):.0f} m")
            if qq(ys, .01) < -2 or qq(ys, .99) > 72:
                malo.append("hay jugadores fuera del campo a lo ancho")
            if 100.0 * off / m > 5:
                malo.append(f"{100.0*off/m:.0f}% de detecciones fuera")
            if malo:
                print("    -> MAL CALIBRADA: " + "; ".join(malo))
                print("       (con el arquero en cámara, x deberia llegar a ~0)")
            else:
                print("    -> calibracion plausible")

        # --- METRICA DEL SINTOMA: pelota amontonada en las areas ---
        nball, inbox = ball_in_penalty_box(path)
        if nball:
            frac = 100.0 * inbox / nball
            esperado = 100.0 * PBOX_AREA_FRAC
            print(f"  PELOTA DENTRO DE UN AREA DE PENAL "
                  f"({nball} frames con pelota)")
            print(f"    {frac:.1f}%  (esperado por superficie {esperado:.1f}%, "
                  f"x{frac/esperado:.1f})")
            if frac > 2 * esperado:
                print("    -> SINTOMA PRESENTE: la pelota se engancha en las "
                      "areas (marca de penal / puntos blancos).")


if __name__ == "__main__":
    main()
