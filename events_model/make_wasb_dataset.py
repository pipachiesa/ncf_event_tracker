"""Convierte los labels de label_ball.py al formato de dataset de WASB-SBDT
(frames PNG numerados + XML estilo CVAT por clip) para fine-tunear wasb_soccer.

WASB (datasets/soccer.py) espera:
    <root>/frames/<video>/<clip>/00001.png ...   (frames consecutivos)
    <root>/annos/<video>/<clip>.xml              (CVAT: x,y,outside,occluded,used_in_game)
y usa secuencias de 3 frames consecutivos, asi que se extraen TODOS los frames
del rango (no solo los anotados); los no anotados van como visibles=0 salvo que
haya label. El fine-tune parte de wasb_soccer_best.pth.tar.

Uso:
    python3 events_model/make_wasb_dataset.py \\
        --video ~/football_data/matches/clip-test/video.mp4 \\
        --labels events_model/dataset/ball_gt/spain-france_ball_labels.csv \\
        --out ~/wasb_ft/soccer --clip spain-france-3min --stride 2
"""
import argparse
import csv
import os
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", default="clip1")
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    lab = {}
    for r in csv.DictReader(open(args.labels)):
        f = int(r["Frame"])
        v = int(r["visible"])
        lab[f] = (float(r["X_img"]), float(r["Y_img"]), v) if v and r["X_img"] != "" \
            else (None, None, 0)
    if not lab:
        raise SystemExit("labels vacio")
    lo, hi = min(lab), max(lab)

    fdir = os.path.expanduser(os.path.join(args.out, "frames", "match", args.clip))
    adir = os.path.expanduser(os.path.join(args.out, "annos", "match"))
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(adir, exist_ok=True)

    cap = cv2.VideoCapture(os.path.expanduser(args.video))
    tracks = []   # (fid, x, y, visible)
    for csv_frame in range(lo, hi + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, (csv_frame - 1) * args.stride)
        ok, fr = cap.read()
        if not ok:
            break
        fid = csv_frame - lo + 1               # 1-indexed en el clip
        cv2.imwrite(os.path.join(fdir, f"{fid:05d}.png"), fr)
        x, y, v = lab.get(csv_frame, (None, None, 0))
        tracks.append((fid, x, y, v))
    cap.release()

    # XML estilo CVAT que datasets/soccer.py sabe leer
    xml = ['<?xml version="1.0"?>', '<annotations>',
           f'  <track id="0" label="ball">']
    for fid, x, y, v in tracks:
        outside = "0" if v else "1"
        occluded = "0" if v else "1"
        px, py = (x, y) if v else (-1.0, -1.0)
        xml.append(f'    <points frame="{fid}" outside="{outside}" '
                   f'occluded="{occluded}" points="{px},{py}">')
        xml.append('      <attribute name="used_in_game">1</attribute>')
        xml.append('    </points>')
    xml += ['  </track>', '</annotations>']
    with open(os.path.join(adir, f"{args.clip}.xml"), "w") as fh:
        fh.write("\n".join(xml))

    vis = sum(1 for _f, _x, _y, v in tracks if v)
    print(f"{len(tracks)} frames extraidos, {vis} con pelota, en {args.out}")
    print(f"config: dataset root_dir={os.path.expanduser(args.out)}, video=match, clip={args.clip}")


if __name__ == "__main__":
    main()
