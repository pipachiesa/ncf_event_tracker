"""Convierte los labels de label_ball.py al formato de dataset de WASB-SBDT
(frames PNG numerados + XML estilo CVAT por clip) para fine-tunear wasb_soccer.

WASB (datasets/soccer.py) espera:
    <root>/frames/<video>/<clip>/00001.png ...   (frames consecutivos)
    <root>/annos/<video>/<clip>.xml              (CVAT: x,y,outside,occluded,used_in_game)
y usa secuencias de 3 frames consecutivos, asi que se extraen TODOS los frames
del rango (no solo los anotados). El fine-tune parte de wasb_soccer_best.pth.tar.

⚠️ BUG ARREGLADO (25-ago): los labels de Felipe estan cada ~3 frames. La version
anterior marcaba TODOS los frames intermedios (no anotados) como visible=0, o sea
"sin pelota". Eran ~1.000 NEGATIVOS FALSOS: la pelota SI esta en esos frames,
solo que no se anoto. Entrenar WASB con eso le enseña a NO ver la pelota. Ahora,
para un frame no anotado que cae entre DOS labels visibles con gap <= MAX_INTERP,
se INTERPOLA la posicion (la pelota es suave en 0,1-0,2 s) y se marca visible.
Un frame etiquetado explicitamente no-visible, o no bracketeado por dos visibles
cercanos, queda outside=1 (negativo real / desconocido).

⚠️ PENDIENTE DE VERIFICAR contra el repo WASB: aca se escribe 00001.png con fid
1-indexado; confirmar si `datasets/soccer.py` indexa desde 0 o desde 1.

Uso:
    python3 events_model/make_wasb_dataset.py \\
        --video ~/football_data/matches/clip-test/video.mp4 \\
        --labels events_model/dataset/ball_gt/spain-france_ball_labels.csv \\
        --out ~/wasb_ft/soccer --clip spain-france-3min --stride 2
"""
import argparse
import bisect
import csv
import os
import cv2

# Gap maximo (en frames CSV) entre dos labels visibles para interpolar los
# frames intermedios. Los labels estan cada 3, asi que 3 cubre el caso normal;
# gaps mas grandes suelen ser la pelota entrando/saliendo de vista -> no
# interpolar (queda outside=1) para no inventar pelota donde no la hay.
MAX_INTERP = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", default="clip1")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-interp", type=int, default=MAX_INTERP,
                    help="gap max entre labels visibles para interpolar (frames CSV)")
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

    vis_frames = sorted(f for f, (x, y, v) in lab.items() if v)

    def dense_label(f):
        """La etiqueta anotada de f, o interpolada entre dos labels visibles
        cercanos, o (None,None,0) si no se puede saber."""
        if f in lab:
            return lab[f]
        i = bisect.bisect_left(vis_frames, f)
        if i == 0 or i == len(vis_frames):
            return (None, None, 0)             # fuera del rango anotado
        a, b = vis_frames[i - 1], vis_frames[i]
        if b - a > args.max_interp:
            return (None, None, 0)             # gap grande -> no inventar pelota
        ax, ay, _ = lab[a]
        bx, by, _ = lab[b]
        t = (f - a) / (b - a)
        return (ax + (bx - ax) * t, ay + (by - ay) * t, 1)

    fdir = os.path.expanduser(os.path.join(args.out, "frames", "match", args.clip))
    adir = os.path.expanduser(os.path.join(args.out, "annos", "match"))
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(adir, exist_ok=True)

    cap = cv2.VideoCapture(os.path.expanduser(args.video))
    tracks = []   # (fid, x, y, visible)
    n_interp = 0
    for csv_frame in range(lo, hi + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, (csv_frame - 1) * args.stride)
        ok, fr = cap.read()
        if not ok:
            break
        fid = csv_frame - lo + 1               # 1-indexed en el clip
        cv2.imwrite(os.path.join(fdir, f"{fid:05d}.png"), fr)
        x, y, v = dense_label(csv_frame)
        if v and csv_frame not in lab:
            n_interp += 1
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
    print(f"{len(tracks)} frames extraidos, {vis} con pelota "
          f"({n_interp} interpolados, {vis - n_interp} anotados), en {args.out}")
    print(f"config: dataset root_dir={os.path.expanduser(args.out)}, video=match, clip={args.clip}")


if __name__ == "__main__":
    main()
