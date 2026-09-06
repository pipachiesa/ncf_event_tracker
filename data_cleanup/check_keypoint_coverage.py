"""Mide el TECHO de cualquier esquema de acumulacion de keypoints.

POR QUE EXISTE ESTE SCRIPT
El primer intento de arreglar la calibracion fue acumular keypoints entre
refrescos y arrastrarlos con el paneo (``KeypointBuffer`` +
``median_image_shift``). MEDIDO en el clip de 3 min:

    Keypoints acumulados al final: 150 puntos cubriendo 20 x 56 m de cancha
    Homografia: 29 aceptadas, 486 descartadas por calidad (94.6% rechazo)

150 puntos / 15 refrescos = 10 por refresco: se acumularon quince copias del
MISMO parche. La razon es geometrica, no un bug: el span se mide sobre las
coordenadas de CANCHA, que son etiquetas fijas del vertice detectado, asi que
solo crece si el modelo ve vertices DISTINTOS en refrescos distintos. Con la
camara paneando ~1,25 px por refresco, en 5 s se movio 19 px y ve los mismos
diez vertices. El arrastre puede estar perfecto y no cambia nada.

QUE MIDE ESTE SCRIPT
Para cada ventana de tiempo, la UNION de vertices detectados y cuanta cancha
cubren. Como las coordenadas de cancha son absolutas, esa union es el **techo**
de lo que cualquier esquema de acumulacion podria lograr con esa ventana, sin
importar que tan bien funcione la compensacion del paneo. Si la ventana de N
segundos no llega a los 30 m que exige ``MIN_KEYPOINT_PITCH_X_CM``, entonces
arreglar ``median_image_shift`` no puede servir: la informacion no esta ahi.

Ademas lista los mejores FRAMES DE REFERENCIA: los que solos ven mucha cancha y
ajustan con poco error. Son los candidatos a ancla si se va por el camino de
resolver una homografia buena una vez y despues estimar solo el movimiento de
camara (pocos grados de libertad) contra ella.

Uso (en Colab, donde estan los modelos):
    python3 data_cleanup/check_keypoint_coverage.py \\
        --video /content/drive/MyDrive/football_analytics/videos/spain-france-test3min.mp4
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Los mismos minimos que exige main.py para aceptar una solucion.
MIN_X_M, MIN_Y_M = 30.0, 15.0


def pct(vals, q):
    if not vals:
        return float("nan")
    v = sorted(vals)
    return v[min(len(v) - 1, int(q * (len(v) - 1)))]


def span_of(mask_int, verts):
    """Span en cancha (m) de los vertices marcados en el bitmask."""
    idx = [i for i in range(len(verts)) if mask_int >> i & 1]
    if len(idx) < 2:
        return 0.0, 0.0, len(idx)
    d = verts[idx]
    return ((d[:, 0].max() - d[:, 0].min()) / 100.0,
            (d[:, 1].max() - d[:, 1].min()) / 100.0,
            len(idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--stride", type=int, default=10,
                    help="frames de VIDEO entre muestras. El default 10 es lo "
                         "que ve main.py: --frame-stride 2 x --homography-every 5")
    ap.add_argument("--pitch-conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="0 = el clip entero")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO
    from pitch_config import PITCH
    from sports.common.view import ViewTransformer
    import supervision as sv
    import main as M

    cfg = PITCH
    verts = np.array(cfg.vertices, dtype=np.float32)

    path = M.resolve_pitch_model_path(M.DEFAULT_PITCH_MODEL)
    if path is None:
        sys.exit("no pude resolver el modelo de keypoints de cancha")
    model = YOLO(path)

    cap = cv2.VideoCapture(os.path.expanduser(args.video))
    if not cap.isOpened():
        sys.exit(f"no pude abrir el video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_expected = nframes // args.stride
    if args.max_samples:
        n_expected = min(n_expected, args.max_samples)

    print(f"video: {os.path.basename(args.video)}")
    print(f"  {nframes} frames @ {fps:g} fps, muestreando cada {args.stride} "
          f"-> {n_expected} muestras ({args.stride/fps:.2f} s entre muestras)")
    print(f"  umbral de confianza: {args.pitch_conf}")
    print(f"  cancha: {cfg.length/100:.0f} x {cfg.width/100:.0f} m, "
          f"{len(verts)} vertices\n")

    try:
        from tqdm import tqdm
        bar = tqdm(total=n_expected, desc="muestreando")
    except Exception:
        bar = None

    samples = []          # (frame_no, bitmask, span_x, span_y, n_kp, reproy_cm)
    fno = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        fno += 1
        if (fno - 1) % args.stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break

        res = model(frame, imgsz=args.imgsz, verbose=False)[0]
        kp = sv.KeyPoints.from_ultralytics(res)
        if kp.xy is None or len(kp.xy) == 0 or kp.confidence is None:
            samples.append((fno, 0, 0.0, 0.0, 0, float("nan")))
        else:
            conf = kp.confidence[0]
            m = conf > args.pitch_conf
            bits = 0
            for i in np.where(m)[0]:
                bits |= 1 << int(i)
            sx, sy, nk = span_of(bits, verts)
            err = float("nan")
            if nk >= 4:
                try:
                    s = kp.xy[0][m].astype(np.float32)
                    d = verts[m].astype(np.float32)
                    tt = ViewTransformer(source=s, target=d)
                    err = float(np.median(np.linalg.norm(
                        tt.transform_points(points=s) - d, axis=1)))
                except Exception:
                    pass
            samples.append((fno, bits, sx, sy, nk, err))

        if bar:
            bar.update(1)
        if args.max_samples and len(samples) >= args.max_samples:
            break
    if bar:
        bar.close()
    cap.release()

    if not samples:
        sys.exit("no se pudo muestrear ningun frame")

    n = len(samples)
    kps = [s[4] for s in samples]
    sxs = [s[2] for s in samples]
    sys_ = [s[3] for s in samples]
    solos = sum(1 for s in samples if s[2] >= MIN_X_M and s[3] >= MIN_Y_M)

    print(f"\n{'='*68}")
    print("POR MUESTRA — lo que ve main.py en UN refresco")
    print(f"{'='*68}")
    print(f"  keypoints    p50 {pct(kps,.5):5.0f}   p90 {pct(kps,.9):5.0f}   "
          f"max {max(kps):5.0f}")
    print(f"  span x (m)   p50 {pct(sxs,.5):5.1f}   p90 {pct(sxs,.9):5.1f}   "
          f"max {max(sxs):5.1f}")
    print(f"  span y (m)   p50 {pct(sys_,.5):5.1f}   p90 {pct(sys_,.9):5.1f}   "
          f"max {max(sys_):5.1f}")
    print(f"  muestras que SOLAS pasan el minimo ({MIN_X_M:.0f} x {MIN_Y_M:.0f} m): "
          f"{solos} de {n} ({100.0*solos/n:.1f}%)")

    # Cuantos vertices distintos aparecen en total. Si es igual al de una
    # muestra tipica, no hay nada que acumular: la camara siempre ve lo mismo.
    total_bits = 0
    for s in samples:
        total_bits |= s[1]
    tot_x, tot_y, tot_n = span_of(total_bits, verts)
    print(f"\n  vertices distintos en TODO el clip: {tot_n} de {len(verts)}   "
          f"(span {tot_x:.0f} x {tot_y:.0f} m)")
    print(f"  vertices en una muestra tipica:      {pct(kps,.5):.0f}")

    # --- EL NUMERO QUE DECIDE ---
    print(f"\n{'='*68}")
    print("TECHO DE LA ACUMULACION — union de vertices en ventanas consecutivas")
    print(f"{'='*68}")
    print("  Es un TECHO: las coordenadas de cancha son absolutas, asi que esto")
    print("  es lo mejor que podria dar cualquier acumulacion con esa ventana,")
    print("  por bien que funcione la compensacion del paneo.\n")
    print(f"  {'ventana':>12} {'muestras':>9} {'span x p50':>11} "
          f"{'span x max':>11} {'% >= 30 m':>10}")

    dt = args.stride / fps
    ventanas_s = [1, 5, 15, 30, 60, 120]
    mejor_ventana = None
    for w_s in ventanas_s:
        k = max(1, int(round(w_s / dt)))
        if k > n:
            continue
        spans = []
        for i in range(0, n - k + 1):
            bits = 0
            for j in range(i, i + k):
                bits |= samples[j][1]
            sx, sy, _ = span_of(bits, verts)
            spans.append(sx)
        ok_frac = 100.0 * sum(1 for s in spans if s >= MIN_X_M) / len(spans)
        print(f"  {w_s:>9} s {k:>9} {pct(spans,.5):>11.1f} "
              f"{max(spans):>11.1f} {ok_frac:>9.0f}%")
        if mejor_ventana is None and ok_frac >= 80:
            mejor_ventana = (w_s, pct(spans, .5))
    print(f"  {'clip entero':>12} {n:>9} {tot_x:>11.1f} {tot_x:>11.1f} "
          f"{100.0 if tot_x >= MIN_X_M else 0.0:>9.0f}%")

    # --- CANDIDATOS A FRAME DE REFERENCIA ---
    print(f"\n{'='*68}")
    print("FRAMES DE REFERENCIA — los que solos ven mucha cancha y ajustan bien")
    print(f"{'='*68}")
    buenos = [s for s in samples if s[4] >= 6 and not np.isnan(s[5])]
    buenos.sort(key=lambda s: (-s[2], s[5]))
    if not buenos:
        print("  ninguno con >= 6 keypoints y ajuste resoluble")
    else:
        print(f"  {'frame':>7} {'t (s)':>7} {'kps':>4} {'span x':>7} "
              f"{'span y':>7} {'reproy (cm)':>12}")
        for s in buenos[:10]:
            print(f"  {s[0]:>7} {s[0]/fps:>7.1f} {s[4]:>4} {s[2]:>7.1f} "
                  f"{s[3]:>7.1f} {s[5]:>12.0f}")

    # --- VEREDICTO ---
    print(f"\n{'='*68}")
    print("VEREDICTO")
    print(f"{'='*68}")
    if tot_x < MIN_X_M:
        print("  ❌ NI EL CLIP ENTERO llega al minimo de cancha. El problema no")
        print("     es la acumulacion sino la DETECCION de keypoints: el modelo")
        print("     nunca ve mas que este parche. Hay que bajar el umbral, usar")
        print("     otro modelo de cancha, o calibrar a mano con puntos fijos.")
    elif mejor_ventana:
        w_s, p50 = mejor_ventana
        k = max(1, int(round(w_s / dt)))
        print(f"  ✅ ACUMULAR SIRVE con ventana de ~{w_s} s ({k} refrescos):")
        print(f"     el 80% de las ventanas llega a {MIN_X_M:.0f} m "
              f"(mediana {p50:.0f} m).")
        print(f"     KEYPOINT_BUFFER_REFRESHES = {k}  (hoy esta en "
              f"{M.KEYPOINT_BUFFER_REFRESHES}).")
        print("     OJO: a esa ventana el arrastre por traslacion acumula error")
        print("     sobre muchos pasos. Verificar el error de reproyeccion.")
    else:
        print("  🔴 NINGUNA VENTANA RAZONABLE alcanza el minimo, pero el clip")
        print(f"     entero si ({tot_x:.0f} m). O sea: la informacion existe pero")
        print("     esta repartida en minutos, no en segundos.")
        print("     -> Camino recomendado: FRAME DE REFERENCIA. Resolver UNA")
        print("        homografia buena con los frames de arriba y despues")
        print("        estimar por refresco solo el movimiento de camara (2-4")
        print("        grados de libertad), que un parche de 20 m si determina.")
        print("        No hay deriva: cada refresco re-ancla contra keypoints")
        print("        absolutos.")


if __name__ == "__main__":
    main()
