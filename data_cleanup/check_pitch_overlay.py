"""Dibuja la cancha SEGUN EL MAPA encima del video. El test absoluto.

POR QUE ESTE SCRIPT ES DISTINTO DE TODOS LOS OTROS CHEQUEOS
``check_homography.py`` mide cosas INDIRECTAS: que las coordenadas no salten,
que los jugadores ocupen un rango plausible, que la pelota no se amontone en
las areas. Todas son proxies, y cada proxy tuvo su falso positivo:

  * un mapa CONGELADO saca nota perfecta en estabilidad mientras pone al
    arquero a 39 m de su arco;
  * "pelota dentro del area" dio 6,1% con un mapa roto (parecia buenisimo) y
    19,7% con otro mapa roto (parecia perfecto);
  * "x: p99 no llega a 110" puede ser falsa alarma si la camara nunca muestra
    la mitad lejana.

Este script no usa proxies: proyecta el modelo de cancha (lineas de gol,
areas, circulo central, punto de penal) sobre el frame de video y lo compara
con las lineas PINTADAS, que estan ahi y no mienten. Si el amarillo cae sobre
el blanco, el mapa esta bien. Si no, se ve exactamente cuanto y para donde
esta corrido.

MEDIDO asi el 18-ago sobre la corrida (5), la mejor hasta ese momento: el
circulo central proyectado caia ~15 m a la izquierda del circulo pintado, y el
punto de penal caia fuera del area. O sea que la homografia seguia MAL a pesar
de que el rechazo habia bajado de 94,6% a 27,6% y de que todos los indicadores
indirectos habian mejorado.

COMO RECUPERA LA HOMOGRAFIA
No hace falta que el CSV la guarde: cada deteccion de jugador trae su posicion
en IMAGEN y en CANCHA, y las dos estan relacionadas por la transformacion que
uso main.py. Con 4 o mas jugadores se recupera exacto (el residuo que imprime
es la verificacion: tiene que dar ~0 cm). Por eso corre sobre CUALQUIER
tracking ya generado, sin GPU ni modelos, y sirve para auditar corridas viejas.

Uso:
    python3 data_cleanup/check_pitch_overlay.py \\
        --tracking-csv ~/football_data/matches/clip-test/tracking.csv \\
        --video        ~/football_data/matches/clip-test/video.mp4 \\
        --frames 317 647 761 1292 \\
        --output /tmp/overlay.jpg
"""

import argparse
import csv
import json
import os

import numpy as np

# Cancha de SoccerPitchConfiguration, en cm.
L, W = 12000.0, 7000.0
PBOX_L, PBOX_W = 2015.0, 4100.0
GBOX_L, GBOX_W = 550.0, 1832.0
PEN_SPOT = 1100.0
CIRCLE_R = 915.0


def _seg(a, b, n=40):
    return [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
            for i in range(n + 1)]


def pitch_lines():
    """Las lineas del modelo, en coordenadas de cancha (cm)."""
    y0, y1 = (W - PBOX_W) / 2, (W + PBOX_W) / 2
    g0, g1 = (W - GBOX_W) / 2, (W + GBOX_W) / 2
    lines = [
        _seg((0, 0), (0, W)), _seg((L, 0), (L, W)),          # lineas de gol
        _seg((0, 0), (L, 0)), _seg((0, W), (L, W)),          # laterales
        _seg((L / 2, 0), (L / 2, W)),                        # linea media
    ]
    for x0 in (0.0, L - PBOX_L):                             # areas de penal
        x1 = x0 + PBOX_L if x0 == 0 else L
        xf = PBOX_L if x0 == 0 else L - PBOX_L
        lines += [_seg((x0 if x0 else 0, y0), (xf, y0)),
                  _seg((xf, y0), (xf, y1)),
                  _seg((xf, y1), (x0 if x0 else 0, y1))]
    for x0 in (0.0, L - GBOX_L):                             # areas chicas
        xf = GBOX_L if x0 == 0 else L - GBOX_L
        lines += [_seg((x0 if x0 else 0, g0), (xf, g0)),
                  _seg((xf, g0), (xf, g1)),
                  _seg((xf, g1), (x0 if x0 else 0, g1))]
    lines.append([(L / 2 + CIRCLE_R * np.cos(t), W / 2 + CIRCLE_R * np.sin(t))
                  for t in np.linspace(0, 2 * np.pi, 72)])
    return lines


def read_frames(path):
    """``frame -> {"pl": [(img_x, img_y, pitch_x, pitch_y)], "ball": (...)}``"""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                px, py = float(r["X_Pitch"]), float(r["Y_Pitch"])
            except (TypeError, ValueError):
                continue
            if px == 0 and py == 0:
                continue
            f = int(r["Frame"])
            d = out.setdefault(f, {"pl": [], "ball": None})
            x1, y1 = float(r["X1"]), float(r["Y1"])
            x2, y2 = float(r["X2"]), float(r["Y2"])
            if r["Object"] == "ball":
                d["ball"] = ((x1 + x2) / 2, (y1 + y2) / 2)
            else:
                # main.py proyecta el BOTTOM_CENTER de la caja (los pies).
                d["pl"].append(((x1 + x2) / 2, y2, px, py))
    return out


def recover_homography(players):
    """La transformacion imagen->cancha que uso main.py, y su residuo en cm."""
    import cv2
    if len(players) < 4:
        return None, None
    src = np.array([[p[0], p[1]] for p in players], dtype=np.float32)
    dst = np.array([[p[2], p[3]] for p in players], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        return None, None
    back = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    return H, float(np.median(np.linalg.norm(back - dst, axis=1)))


def read_stride(tracking, default=1):
    meta = tracking.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(meta):
        with open(meta) as fh:
            return int(json.load(fh).get("frame_stride") or default)
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-csv", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, nargs="+",
                    help="frames del CSV (no del video). Default: 4 repartidos")
    ap.add_argument("--output", default="overlay.jpg")
    ap.add_argument("--width", type=int, default=960)
    args = ap.parse_args()

    import cv2

    tracking = os.path.expanduser(args.tracking_csv)
    rows = read_frames(tracking)
    if not rows:
        raise SystemExit("el CSV no tiene detecciones con coordenadas de cancha")
    stride = read_stride(tracking)

    frames = args.frames
    if not frames:
        usable = sorted(f for f, d in rows.items() if len(d["pl"]) >= 6)
        frames = [usable[int(q * (len(usable) - 1))] for q in (.1, .4, .6, .9)]

    cap = cv2.VideoCapture(os.path.expanduser(args.video))
    if not cap.isOpened():
        raise SystemExit(f"no pude abrir el video: {args.video}")

    lines = pitch_lines()
    tiles = []
    for f in frames:
        if f not in rows:
            print(f"frame {f}: sin detecciones")
            continue
        H, err = recover_homography(rows[f]["pl"])
        if H is None:
            print(f"frame {f}: menos de 4 jugadores, no se puede recuperar")
            continue
        try:
            Hi = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, (f - 1) * stride)
        ok, frame = cap.read()
        if not ok:
            print(f"frame {f}: no se pudo leer del video")
            continue

        for ln in lines:
            p = cv2.perspectiveTransform(
                np.array(ln, dtype=np.float32).reshape(-1, 1, 2), Hi).reshape(-1, 2)
            for a, b in zip(p, p[1:]):
                if not (np.isfinite(a).all() and np.isfinite(b).all()):
                    continue
                if max(abs(a[0]), abs(b[0]), abs(a[1]), abs(b[1])) > 1e5:
                    continue
                cv2.line(frame, tuple(np.int32(a)), tuple(np.int32(b)),
                         (0, 255, 255), 2)
        for x in (PEN_SPOT, L - PEN_SPOT):
            s = cv2.perspectiveTransform(
                np.array([[x, W / 2]], dtype=np.float32).reshape(-1, 1, 2),
                Hi).reshape(2)
            if np.isfinite(s).all() and abs(s[0]) < 1e5 and abs(s[1]) < 1e5:
                cv2.drawMarker(frame, tuple(np.int32(s)), (255, 0, 255),
                               cv2.MARKER_CROSS, 40, 3)
        if rows[f]["ball"]:
            bx, by = rows[f]["ball"]
            cv2.circle(frame, (int(bx), int(by)), 30, (0, 0, 255), 2)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(frame, f"frame {f}  residuo {err:.0f} cm   "
                    f"AMARILLO=cancha segun el mapa  MAGENTA=punto penal  "
                    f"ROJO=pelota", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 1, cv2.LINE_AA)
        h = int(frame.shape[0] * args.width / frame.shape[1])
        tiles.append(cv2.resize(frame, (args.width, h)))
    cap.release()

    if not tiles:
        raise SystemExit("no se pudo renderizar ningun frame")
    if len(tiles) > 1:
        rows_img = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - 1, 2)]
        sheet = np.vstack(rows_img) if len(rows_img) > 1 else rows_img[0]
    else:
        sheet = tiles[0]
    out = os.path.expanduser(args.output)
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"escrito: {out}")
    print("COMO LEERLO: si el amarillo cae sobre las lineas pintadas, el mapa")
    print("esta bien. Mira sobre todo el CIRCULO CENTRAL, que es el punto mas")
    print("lejano del area donde se detectan los keypoints y donde el error se")
    print("nota primero. El residuo tiene que dar ~0 cm: es la verificacion de")
    print("que se recupero la MISMA transformacion que uso main.py.")


if __name__ == "__main__":
    main()
