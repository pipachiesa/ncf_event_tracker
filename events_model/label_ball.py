"""Anotacion rapida de la pelota para fine-tunear un detector temporal (WASB).

FLUJO (minimiza anotacion, ver deteccion-balon-alternativas en memoria):
  1. Pseudo-labels GRATIS: donde el tracking actual ya trackea bien la pelota
     (se mueve, no es la marca), su posicion es label sin anotar. Este script
     las PRE-CARGA y solo pide confirmar/corregir los huecos.
  2. Anotacion manual: en cada frame sin pelota (o dudoso), click = pelota ahi;
     tecla 'n' = no visible; 'espacio' = aceptar la pre-cargada; flechas navegan.

Genera `<match>_ball_labels.csv` (Frame, X_img, Y_img, visible) que alimenta el
dataset de fine-tune de WASB (formato: frames extraidos + csv de posiciones).

Uso:
    python3 events_model/label_ball.py \\
        --video ~/football_data/matches/clip-test/video.mp4 \\
        --tracking ~/football_data/matches/clip-test-new/tracking_recal.csv \\
        --out events_model/dataset/ball_gt/clip-test_ball_labels.csv \\
        --every 3        # anotar 1 de cada N frames (WASB usa secuencias)
"""

import argparse
import csv
import os
import cv2


def load_ball(tracking):
    """{frame_csv: (x_img, y_img)} de la pelota ya trackeada (pseudo-label)."""
    out = {}
    with open(tracking) as fh:
        for r in csv.DictReader(fh):
            if r["Object"] != "ball":
                continue
            x1, y1, x2, y2 = (float(r["X1"]), float(r["Y1"]),
                              float(r["X2"]), float(r["Y2"]))
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                continue
            out[int(r["Frame"])] = ((x1 + x2) / 2, (y1 + y2) / 2)
    return out


def read_stride(tracking):
    meta = tracking.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(meta):
        import json
        return int(json.load(open(meta)).get("frame_stride") or 1)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracking", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--start", type=int, default=1)
    args = ap.parse_args()

    stride = read_stride(args.tracking)
    ball = load_ball(args.tracking)
    cap = cv2.VideoCapture(os.path.expanduser(args.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // stride

    labels = {}
    if os.path.exists(args.out):
        for r in csv.DictReader(open(args.out)):
            labels[int(r["Frame"])] = (r["X_img"], r["Y_img"], int(r["visible"]))

    state = {"click": None}

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    win = "anotar pelota  [click=pelota, espacio=aceptar pre-carga, n=no visible, "\
          "flechas=navegar, s=guardar, q=salir]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    frames = list(range(args.start, total + 1, args.every))
    i = 0
    while 0 <= i < len(frames):
        cf = frames[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, (cf - 1) * stride)
        ok, fr = cap.read()
        if not ok:
            i += 1
            continue
        pre = ball.get(cf)
        disp = fr.copy()
        if cf in labels:
            xx, yy, vis = labels[cf]
            if vis:
                cv2.circle(disp, (int(float(xx)), int(float(yy))), 12, (0, 255, 0), 2)
            txt, col = ("LABEL" + ("" if vis else " (no visible)"), (0, 255, 0))
        elif pre:
            cv2.circle(disp, (int(pre[0]), int(pre[1])), 12, (0, 165, 255), 2)
            txt, col = ("pre-carga (espacio acepta)", (0, 165, 255))
        else:
            txt, col = ("SIN pelota: click o 'n'", (0, 0, 255))
        n_done = len(labels)
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(disp, f"f{cf}  {i+1}/{len(frames)}  anotados {n_done}   {txt}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)
        cv2.imshow(win, disp)
        state["click"] = None
        k = cv2.waitKey(30) & 0xFF
        if state["click"]:
            labels[cf] = (state["click"][0], state["click"][1], 1)
            i += 1
        elif k == ord(' '):
            if pre:
                labels[cf] = (round(pre[0], 1), round(pre[1], 1), 1)
            i += 1
        elif k == ord('n'):
            labels[cf] = ("", "", 0)
            i += 1
        elif k == 83 or k == ord('.'):   # flecha derecha
            i += 1
        elif k == 81 or k == ord(','):   # flecha izquierda
            i = max(0, i - 1)
        elif k == ord('s'):
            _save(args.out, labels)
            print(f"guardado ({len(labels)})")
        elif k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    _save(args.out, labels)
    print(f"guardado {args.out} ({len(labels)} labels, "
          f"{sum(1 for v in labels.values() if v[2])} visibles)")


def _save(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Frame", "X_img", "Y_img", "visible"])
        for f in sorted(labels):
            x, y, v = labels[f]
            w.writerow([f, x, y, v])


if __name__ == "__main__":
    main()
