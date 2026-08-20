"""Error de calibracion EN METROS contra un ground truth anotado a mano.

POR QUE HACE FALTA
Durante semanas se midio la calibracion con proxies (que las coordenadas no
salten, que los jugadores ocupen un rango plausible, que la pelota no se
amontone en las areas) y todos tuvieron su falso positivo. ``check_pitch_overlay``
resolvio eso mostrando el error, pero a ojo: sirve para ver que algo esta mal,
no para comparar dos intentos.

Esto lo convierte en un numero. Sobre un frame con la cancha anotada a mano se
compara donde cae cada punto segun el mapa del pipeline contra donde cae de
verdad. Corre local, en segundos, sobre cualquier tracking ya generado.

MEDIDO sobre la corrida (5) del clip (la mejor hasta el 20-ago):

    landmark                        segun GT     segun pipeline   error
    arco (linea de gol, centro)       (0,35)          (2,21)      14,1 m
    punto de penal                   (11,35)         (20,24)      13,8 m
    centro del circulo central       (60,35)        (102,39)      42,0 m
    error sobre la cancha visible: p50 26,9 m, p90 63,9 m

El error CRECE con la distancia al area de penal: 14 m cerca del arco, 42 m en
el circulo central. Es la firma exacta de una homografia ajustada a un parche
de 20 m y extrapolada al resto.

EL GROUND TRUTH NO VA AL PIPELINE
Son 6 puntos leidos a mano sobre UN frame de UN clip, y sirven solo de regla.
El detector de keypoints sigue siendo el que trabaja en produccion; por eso el
sistema puede cambiar de camara y este archivo no.

Uso:
    python3 data_cleanup/check_calibration_error.py \\
        --tracking-csv ~/football_data/matches/clip-test/tracking.csv
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LANDMARKS = [
    ("arco (linea de gol, centro)", (0, 3500)),
    ("punto de penal", (1100, 3500)),
    ("borde del area", (2015, 3500)),
    ("esquina lejana del area", (2015, 1450)),
    ("esquina cercana del area", (2015, 5550)),
    ("centro del circulo central", (6000, 3500)),
]
DEFAULT_GT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_gt")


def load_gt(path):
    import cv2
    with open(path) as fh:
        gt = json.load(fh)
    src = np.array([p["img"] for p in gt["puntos"]], dtype=np.float32)
    dst = np.array([p["pitch"] for p in gt["puntos"]], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        raise SystemExit(f"no pude resolver el ground truth de {path}")
    back = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    res = float(np.median(np.linalg.norm(back - dst, axis=1)))
    return gt, H, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-csv", required=True)
    ap.add_argument("--gt", default=None,
                    help="json de ground truth; por defecto todos los de calib_gt/")
    args = ap.parse_args()

    import cv2
    from check_pitch_overlay import read_frames, recover_homography

    gts = [args.gt] if args.gt else sorted(glob.glob(os.path.join(DEFAULT_GT, "*.json")))
    if not gts:
        raise SystemExit(f"no hay ground truth en {DEFAULT_GT}")

    rows = read_frames(os.path.expanduser(args.tracking_csv))
    print(f"=== {os.path.basename(args.tracking_csv)} ===")

    for gt_path in gts:
        gt, Hgt, res = load_gt(gt_path)
        frame = int(gt["csv_frame"])
        print(f"\nground truth: {os.path.basename(gt_path)}  frame {frame}  "
              f"(residuo propio {res:.0f} cm)")
        if frame not in rows:
            print("  el CSV no tiene detecciones en ese frame")
            continue
        Hp, pres = recover_homography(rows[frame]["pl"])
        if Hp is None:
            print("  menos de 4 jugadores: no se puede recuperar el mapa")
            continue
        print(f"  mapa del pipeline recuperado con residuo {pres:.0f} cm "
              f"({len(rows[frame]['pl'])} jugadores)\n")

        Hgi = np.linalg.inv(Hgt)
        print(f"  {'landmark':<32}{'real':>12}{'pipeline':>14}{'error':>10}")
        for name, (px, py) in LANDMARKS:
            img = cv2.perspectiveTransform(
                np.array([[px, py]], dtype=np.float32).reshape(-1, 1, 2), Hgi).reshape(2)
            if not np.isfinite(img).all():
                continue
            got = cv2.perspectiveTransform(
                img.reshape(-1, 1, 2).astype(np.float32), Hp).reshape(2)
            err = float(np.linalg.norm(got - np.array([px, py]))) / 100.0
            print(f"  {name:<32}{f'({px/100:.0f},{py/100:.0f})':>12}"
                  f"{f'({got[0]/100:.0f},{got[1]/100:.0f})':>14}{err:>8.1f} m")

        # Error sobre toda la parte de cancha que se ve en el frame.
        gx, gy = np.meshgrid(np.linspace(0, 12000, 25), np.linspace(0, 7000, 15))
        P = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
        img = cv2.perspectiveTransform(P.reshape(-1, 1, 2), Hgi).reshape(-1, 2)
        ok = (np.isfinite(img).all(1) & (img[:, 0] > 0) & (img[:, 0] < 1920)
              & (img[:, 1] > 0) & (img[:, 1] < 1080))
        if ok.sum() >= 4:
            got = cv2.perspectiveTransform(
                img[ok].reshape(-1, 1, 2), Hp).reshape(-1, 2)
            e = np.linalg.norm(got - P[ok], axis=1) / 100.0
            print(f"\n  sobre {int(ok.sum())} puntos de la cancha visibles en el frame:")
            print(f"    error  p50 {np.median(e):5.1f} m   p90 {np.percentile(e,90):5.1f} m"
                  f"   max {e.max():5.1f} m")
            print(f"    OBJETIVO: p50 < 1 m. Un mapa sano tiene que dar decimas.")


if __name__ == "__main__":
    main()
