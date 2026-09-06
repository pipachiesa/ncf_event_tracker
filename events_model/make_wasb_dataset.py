"""Convierte los labels de label_ball.py al formato de dataset de WASB-SBDT
(frames PNG numerados + XML estilo CVAT por clip) para fine-tunear wasb_soccer.

WASB (datasets/soccer.py) espera (VERIFICADO contra el repo nttcom/WASB-SBDT):
    <root>/frames/<video>/00000.png ...   (0-INDEXED, frames consecutivos, plano)
    <root>/annos/<video>.xml              (CVAT: x,y,outside,occluded,used_in_game)
donde <video> es el nombre del clip (NO hay nivel extra anidado). El loader arma
`'{:05d}.png'.format(fid)` con fid desde 0 y matchea el atributo frame= del XML.
Usa secuencias de 3 frames consecutivos, asi que se extraen TODOS los frames del
rango (no solo los anotados). El fine-tune parte de wasb_soccer_best.pth.tar.

Solo exporta rangos completamente anotados. El loader XML de WASB convierte
frames sin anotacion en negativos, por eso los labels ralos se rechazan ANTES
de extraer imagenes. Para usarlos sin inventar targets: wasb_sparse.py.

Nota: el config del notebook de WASB debe usar frame_dirname="frames",
anno_dirname="annos" y videos=[<clip>] para que las rutas coincidan.

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
    ap.add_argument("--max-interp", type=int, default=0,
                    help="obsoleto: no se permite interpolar etiquetas de entrenamiento")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0=todo el rango; >0 recorta a los primeros N frames (smoke test)")
    args = ap.parse_args()
    if args.stride < 1 or args.max_frames < 0 or args.max_interp != 0:
        ap.error("stride debe ser positivo; max-frames >=0; interpolacion deshabilitada")

    lab = {}
    for r in csv.DictReader(open(args.labels)):
        f = int(r["Frame"])
        v = int(r["visible"])
        lab[f] = (float(r["X_img"]), float(r["Y_img"]), v) if v and r["X_img"] != "" \
            else (None, None, 0)
    if not lab:
        raise SystemExit("labels vacio")
    lo, hi = min(lab), max(lab)

    if args.max_frames:
        hi = min(hi, lo + args.max_frames - 1)

    # WASB's XML loader treats absent/unknown targets as negative. Do not
    # manufacture supervision; sparse labels belong in wasb_sparse.py.
    missing = sum(f not in lab for f in range(lo, hi + 1))
    if missing:
        raise SystemExit(
            f"Exportacion cancelada: {missing} frames sin etiqueta real. "
            "El loader XML los convierte en negativos falsos. "
            "Usar wasb_sparse.py (supervision central sin interpolacion) "
            "o anotar todos los frames del rango.")

    fdir = os.path.expanduser(os.path.join(args.out, "frames", args.clip))
    adir = os.path.expanduser(os.path.join(args.out, "annos"))
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(adir, exist_ok=True)

    cap = cv2.VideoCapture(os.path.expanduser(args.video))
    tracks = []   # (fid, x, y, visible)
    for csv_frame in range(lo, hi + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, (csv_frame - 1) * args.stride)
        ok, fr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"No se pudo leer frame {csv_frame}")
        fid = csv_frame - lo                   # 0-indexed (loader WASB es 0-based)
        if not cv2.imwrite(os.path.join(fdir, f"{fid:05d}.png"), fr):
            raise RuntimeError("No se pudo escribir imagen")
        x, y, v = lab[csv_frame]
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
    print(f"{len(tracks)} frames extraidos ({tracks[0][0]:05d}.png..{tracks[-1][0]:05d}.png), "
          f"{vis} con pelota (todos anotados), en {args.out}")
    print(f"config WASB: root_dir={os.path.expanduser(args.out)}  videos=['{args.clip}']  "
          f"frame_dirname='frames'  anno_dirname='annos'")


if __name__ == "__main__":
    main()
