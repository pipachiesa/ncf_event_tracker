"""Diagnostica la CALIBRACION de la homografia (no su estabilidad).

POR QUE HACE FALTA
``check_homography.py`` mide que las coordenadas no SALTEN entre frames. Una
transformacion constante y completamente equivocada pasa ese test con nota
perfecta. MEDIDO en el clip de 3 min ya arreglado: saltos > 1 m en solo el
0,33% de los frames (estabilidad excelente) pero el arquero, parado dentro de
su arco, proyecta a **x = 39,4 m** cuando la linea de gol es x = 0, y en 3
minutos de juego NINGUN jugador aparece nunca en los primeros 35 metros del
campo. La homografia esta corrida ~35-40 m de forma sistematica.

QUE MIRA ESTE SCRIPT
El emparejamiento entre los keypoints que detecta el modelo y los vertices de
``SoccerPitchConfiguration``. ``build_pitch_transformer`` hace:

    mask      = conf > min_conf
    frame_pts = key_points.xy[0][mask]
    pitch_pts = np.array(SoccerPitchConfiguration().vertices)[mask]

o sea ASUME que el keypoint i del modelo es el vertice i de la configuracion.
Si el modelo usa otro orden, u otra geometria de cancha, la homografia sale
auto-consistente pero apuntando al lugar equivocado -- exactamente el sintoma.

QUE IMPRIME, por frame:
  * cuantos keypoints superan el umbral y CUALES (sus indices)
  * el error de reproyeccion de la solucion
  * donde proyectan los cuatro corners de la imagen (deberian caer dentro o
    cerca de una cancha de 120x70)
  * donde proyecta el punto medio del borde inferior de cada caja de jugador

Uso (en Colab, donde estan los modelos):
    python3 data_cleanup/check_pitch_keypoints.py \\
        --video ~/football_data/matches/clip-test/video.mp4 \\
        --frames 1 500 1107
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[1],
                    help="frames de VIDEO a inspeccionar")
    ap.add_argument("--pitch-conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO
    from sports.configs.soccer import SoccerPitchConfiguration
    from sports.common.view import ViewTransformer
    import supervision as sv
    import main as M

    cfg = SoccerPitchConfiguration()
    verts = np.array(cfg.vertices, dtype=np.float32)
    print(f"configuracion de cancha: {cfg.length} x {cfg.width} cm "
          f"({cfg.length/100:.0f} x {cfg.width/100:.0f} m), "
          f"{len(verts)} vertices")
    print(f"  vertice 0  = {verts[0]}")
    print(f"  vertice -1 = {verts[-1]}")
    print(f"  rango x {verts[:,0].min():.0f}-{verts[:,0].max():.0f} cm, "
          f"y {verts[:,1].min():.0f}-{verts[:,1].max():.0f} cm\n")

    # El resolver del modelo de KEYPOINTS es distinto del de jugadores/pelota:
    # 'football-field' se baja de otro repo del Hub.
    path = M.resolve_pitch_model_path(M.DEFAULT_PITCH_MODEL)
    if path is None:
        sys.exit("no pude resolver el modelo de keypoints de cancha")
    model = YOLO(path)

    cap = cv2.VideoCapture(os.path.expanduser(args.video))
    if not cap.isOpened():
        sys.exit(f"no pude abrir el video: {args.video}")
    h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    for fno in args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fno - 1))
        ok, frame = cap.read()
        if not ok:
            print(f"frame {fno}: no se pudo leer")
            continue
        res = model(frame, imgsz=args.imgsz, verbose=False)[0]
        kp = sv.KeyPoints.from_ultralytics(res)
        if kp.xy is None or len(kp.xy) == 0 or kp.confidence is None:
            print(f"\n=== frame {fno}: sin keypoints ===")
            continue
        conf = kp.confidence[0]
        mask = conf > args.pitch_conf
        idx = np.where(mask)[0]
        print(f"\n=== frame {fno} ===")
        print(f"  keypoints con conf > {args.pitch_conf}: {len(idx)} de {len(conf)}")
        print(f"  indices usados: {list(idx)}")
        if len(idx) < 4:
            print("  insuficientes para resolver")
            continue

        src = kp.xy[0][mask].astype(np.float32)
        dst = verts[mask].astype(np.float32)
        for i, s, d in zip(idx, src, dst):
            print(f"    kp {i:>2}  imagen ({s[0]:7.1f},{s[1]:7.1f})  ->  "
                  f"vertice ({d[0]:7.0f},{d[1]:7.0f}) cm")

        t = ViewTransformer(source=src, target=dst)
        back = t.transform_points(points=src)
        err = np.linalg.norm(back - dst, axis=1)
        print(f"  error de reproyeccion: mediana {np.median(err):.0f} cm, "
              f"max {err.max():.0f} cm")
        print("  (si es chico pero los jugadores caen mal, el problema es el "
              "EMPAREJAMIENTO, no el ajuste)")

        corners = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]],
                           dtype=np.float32)
        pc = t.transform_points(points=corners)
        print("  esquinas de la imagen proyectadas (cm):")
        for name, p in zip(("sup-izq", "sup-der", "inf-der", "inf-izq"), pc):
            print(f"    {name}: ({p[0]:9.0f},{p[1]:9.0f})")


if __name__ == "__main__":
    main()
