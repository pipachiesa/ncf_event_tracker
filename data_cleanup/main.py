"""
Command-line tracking entry point.

This is a script version of ``track/track.ipynb``, using **ultralytics YOLOv8**
for detection (no Roboflow API key required). It runs YOLOv8 over a video to
detect players and the ball, and writes raw tracking data (player & ball pitch
coordinates per frame) to a CSV that the rest of the pipeline (``data_cleanup``
cleaning and ``event_generation``) can consume.

Example:
    python data_cleanup/main.py \\
        --video ./track/footage/2e57b9_0.mp4 \\
        --output ./track/output

The CSV is written to ``<output>/<video name>.csv`` in the "raw" format read by
``Match.import_raw_data``.

Note: a generic COCO ``yolov8n.pt`` detects people ("player") and the
"sports ball" but has no pitch keypoint model, so pitch coordinates are taken
directly from image space (each detection's bottom-centre pixel, normalised to
the standard pitch dimensions) rather than via a homography. The coordinates are
therefore in camera perspective and approximate. Point ``--player-model`` /
``--ball-model`` at a football-trained ``.pt`` for better detections.

Ball detection notes:
    * The ball is a tiny, fast object that the nano COCO model misses
      constantly. By default ``--ball-model football`` downloads a
      football-specific YOLOv8 checkpoint trained on broadcast footage (classes
      ``ball/goalkeeper/player/referee``) from the Hugging Face Hub -- no API key
      required -- and falls back to ``yolov8m.pt`` (medium, far better small
      object recall than nano) if that download is unavailable.
    * Ball confidence defaults to a low ``0.15`` so faint/blurred balls are not
      discarded.
    * Short detection gaps are linearly interpolated (see ``interpolate_ball``)
      so a few missed frames don't reset possession downstream.
"""

import argparse
import json
import os

import numpy as np
from tqdm import tqdm


def _patch_torch_load_for_ultralytics():
    """Allow ultralytics ``.pt`` checkpoints to load under PyTorch >= 2.6.

    PyTorch 2.6 flipped ``torch.load``'s default to ``weights_only=True``, which
    refuses to unpickle the ultralytics model globals (``DetectionModel`` etc.)
    and breaks ``YOLO(weights)`` on older ultralytics releases. Our checkpoints
    come from trusted sources (official ultralytics weights and a public
    football model on the HF Hub), so we restore the full-unpickle behaviour.
    """
    try:
        import torch
    except Exception:
        return
    if getattr(torch.load, "_ultralytics_patched", False):
        return
    _orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    _load._ultralytics_patched = True
    torch.load = _load


_patch_torch_load_for_ultralytics()

# Models. The sentinel ``"football"`` resolves (in ``resolve_model_path``) to a
# football-trained YOLOv8 checkpoint downloaded from the Hugging Face Hub
# (classes: ball/goalkeeper/player/referee), with a fallback to a stock
# ultralytics model. Pass any ``.pt`` path to override.
#
# Using the football model for BOTH players and the ball gives far more
# consistent player detection on broadcast footage than the generic COCO
# ``yolov8n.pt`` -- which fragments tracks badly (thousands of short-lived IDs)
# and can't separate players from referees. When both resolve to the same
# weights we run a single inference per frame instead of two.
DEFAULT_PLAYER_MODEL = "football"
FALLBACK_PLAYER_MODEL = "yolov8n.pt"   # generic COCO "person" detector
DEFAULT_BALL_MODEL = "football"
FALLBACK_BALL_MODEL = "yolov8m.pt"     # medium COCO model: best small-object recall

# Public, football-trained YOLOv8 weights (Roboflow "football-players-detection"
# dataset, broadcast footage; classes: ball/goalkeeper/player/referee).
# ``best.pt`` is a standard ultralytics checkpoint, downloadable without an API
# key. mAP@0.5 ~0.785. Loads directly via ``YOLO(path)``.
FOOTBALL_MODEL_REPO = "uisikdag/yolo-v8-football-players-detection"
FOOTBALL_MODEL_FILE = "best.pt"

# The ball is small and easily missed; keep its confidence low.
DEFAULT_BALL_CONF = 0.10
DEFAULT_PLAYER_CONF = 0.3

# Inference resolution. The ball is tiny, so detecting it at higher resolution
# (e.g. 1280 instead of the 640 default) recovers many small-object detections
# the model otherwise misses. Higher = better ball recall but slower.
DEFAULT_IMGSZ = 1280

# Pitch keypoint model (YOLOv8-pose, 32 keypoints matching SoccerPitchConfiguration).
# Enables a real image->pitch homography so event coordinates are true pitch
# positions instead of a naive camera-perspective scaling. ``"football-field"``
# downloads the weights from the HF Hub (no API key); ``"none"`` disables
# homography and falls back to image-space coordinates.
DEFAULT_PITCH_MODEL = "football-field"
PITCH_MODEL_REPO = "martinjolif/yolo-football-pitch-detection"
PITCH_MODEL_FILE = "yolo-football-pitch-detection.pt"
# Minimum keypoint confidence to use a landmark in the homography, and minimum
# number of confident landmarks needed to solve one.
DEFAULT_PITCH_CONF = 0.5
# MEDIDO (spain-france, 89.027 frames): con el minimo de 4 puntos, en el 6,4% de
# los frames TODO el plantel salta mas de 1 m de golpe, y en el 1,7% mas de
# 10 m. El 93,6% de esos saltos cae en frames multiplo de ``homography_every``,
# o sea justo cuando se re-estima la transformacion -- no son los jugadores
# moviendose, es el sistema de coordenadas. Cuatro puntos son el minimo EXACTO
# para resolver una homografia: no sobra ninguno, asi que el error de
# reproyeccion es cero por construccion y es imposible saber si la solucion es
# buena. Con 6 hay redundancia y el error se vuelve medible.
MIN_HOMOGRAPHY_POINTS = 6
# Error de reproyeccion maximo tolerado (cm): se proyectan los propios keypoints
# y se comparan con su posicion conocida en la cancha.
MAX_REPROJECTION_CM = 300.0
# Los keypoints tienen que estar REPARTIDOS. Cuatro puntos amontonados en una
# esquina resuelven una homografia perfecta localmente y disparatada en el resto
# del campo, que es de donde salen los errores de 25 m. Se exige que la caja que
# los contiene cubra algo de la imagen y que no esten casi alineados.
MIN_KEYPOINT_SPREAD = 0.15   # fraccion del ancho/alto de imagen
MIN_KEYPOINT_ASPECT = 0.08   # razon entre los dos ejes principales
# ⚠️ Los dos de arriba miden el spread EN LA IMAGEN, y eso NO alcanza.
# MEDIDO con check_pitch_keypoints.py: en 3 de 4 frames muestreados los unicos
# keypoints confiables eran los indices 0,1,2,6,7,8,9,10,11,12 -- TODOS del area
# de penal izquierda, o sea x ∈ {0, 550, 1100, 2015} cm: un cluster de 20 m
# sobre una cancha de 120. En la IMAGEN ocupaban de x=126 a x=888 px, asi que
# pasaban el filtro sin problema. El error de reproyeccion daba 36-93 cm
# (excelente) porque el ajuste es bueno CERCA de esos puntos; despues extrapola
# al resto de la pantalla y se va a cientos de metros: las esquinas de la imagen
# proyectaban a (-17768,-14706) y (2584, 62716) cm.
# La homografia hay que anclarla a puntos repartidos por la CANCHA, no por la
# pantalla.
MIN_KEYPOINT_PITCH_X_CM = 3000.0   # 30 m de los 120 de largo
MIN_KEYPOINT_PITCH_Y_CM = 1500.0   # 15 m de los 70 de ancho
# Guarda ANTI-CATASTROFE, no control de continuidad. Ver la advertencia:
#
# ⚠️ UNA VERSION ANTERIOR PUSO ESTO EN 250 cm Y ROMPIO TODO. Comparaba a donde
# proyectan LOS MISMOS PIXELES con la transformacion vieja y la nueva, y
# rechazaba la nueva si el resultado difería. Pero cuando la camara PANEA, el
# mismo pixel corresponde a otro punto del mundo: la transformacion TIENE que
# cambiar. El filtro castigaba el comportamiento correcto. MEDIDO en el clip de
# 3 min: 19 homografias aceptadas, 84 descartadas por calidad y 437 por "salto"
# -- 96,5% de rechazo, o sea el clip entero corriendo con un mapa calculado al
# principio y CONGELADO. Un mapa congelado no puede saltar, asi que
# `check_homography.py` daba 0,33% (excelente) mientras el arquero proyectaba a
# 39 m de su arco. Optimizar la metrica de estabilidad premia justamente el
# fallo. Estabilidad y CALIBRACION son cosas distintas: medir las dos.
#
# Este umbral queda solo para descartar soluciones absurdas (decenas de metros),
# que ya deberian caer antes por spread y error de reproyeccion.
MAX_HOMOGRAPHY_JUMP_CM = 1500.0

# --- SUAVIZADO TEMPORAL DE LA HOMOGRAFIA ---
# MEDIDO: los filtros de calidad (6 puntos, spread, error de reproyeccion) NO
# alcanzan -- frames con salto > 1 m 9,12% -> 8,71%, casi nada. El temblor es
# inherente a re-estimar la transformacion desde cero en cada refresco: los
# keypoints tienen un par de pixeles de ruido y cerca del horizonte eso son
# metros de cancha. Rechazar soluciones tampoco sirve (congela el mapa y arruina
# la calibracion, ver MAX_HOMOGRAPHY_JUMP_CM).
#
# La cámara se mueve de forma CONTINUA, asi que la transformacion correcta varia
# suave y el ruido es de alta frecuencia: un promedio movil lo corta sin
# congelar nada. En vez de promediar matrices (numericamente feo) se promedian
# las POSICIONES DE CANCHA a las que proyectan cuatro puntos fijos de la imagen,
# y se reconstruye la transformacion desde ahi.
#
# Con alpha 0.35 la ventana efectiva son ~6 refrescos = 30 frames = 2 s. El
# paneo medido es de 0,25 px/frame, o sea ~7 px de retraso: despreciable frente
# a los metros de ruido que elimina.
HOMOGRAPHY_SMOOTH_ALPHA = 0.35
# Recompute the homography every N frames (the camera pans smoothly, so we reuse
# the last transform in between to avoid a keypoint pass on every frame).
DEFAULT_HOMOGRAPHY_EVERY = 5

# Pitch keypoints are large features and gain nothing from very high resolution,
# so the pitch model is capped here even when players/ball run at a higher imgsz
# (keeps precision where it matters without paying the cost where it doesn't).
DEFAULT_PITCH_IMGSZ = 1280

# Linearly interpolate ball position across detection gaps no longer than this
# many frames (longer gaps stay empty). Prevents possession from resetting on
# every missed detection.
# Raised from 15 once the continuity gate started rejecting bad detections:
# the gate trades coverage for trustworthiness (ball presence 87% -> 74%), and
# interpolation is the safe way to buy that coverage back because a linear fill
# is SMOOTH and cannot reintroduce teleports. Measured on spain-france at
# stride 2 (so the effective gap is half this): ball presence 74.4% -> 90.1%,
# passes 478 -> 594, set pieces 131 -> 125 (slightly better, not worse).
BALL_INTERP_MAX_GAP = 50

# How many ball candidates per frame to keep for the global trajectory solver
# (ball_viterbi.py). The greedy in-loop gate keeps one; these are written to a
# sidecar so the whole path can be re-optimised offline.
BALL_CANDIDATES_PER_FRAME = 4

# Ball-candidate continuity (see ``_pick_ball``). A powerful shot peaks around
# 35 m/s; above that no real ball is moving, so a candidate that far from the
# last known position is a different white object (penalty spot, line, crowd).
MAX_BALL_SPEED_MS = 40.0
# The gate compares distances in PITCH METRES (see _pick_ball), so no pixel
# scale and no slack term are involved. Two earlier attempts failed here:
# a 120 px "jitter slack" was ~6.6 m of pitch and by itself licensed ~95 m/s
# jumps (impossible moves only fell 26.3% -> 17.6%), and gating in pixel space
# stalled at 12.1% because perspective makes a pixel worth far more metres at
# the far touchline. Sweep on psg-bayern in pitch space: 40 m/s keeps ~57% of
# detections at ~2% impossible; 30 m/s reaches 0% but costs 8 more points of
# coverage, which is not worth it (interpolate_ball bridges short gaps anyway).
DEFAULT_TRACK_FPS = 15.0    # only a fallback; the real effective fps is passed in

# --- Deteccion de pelota por RECORTE de alta resolucion ---
# A 1080p la pelota mide ~9 px (medido: mediana de las cajas detectadas), justo
# en el limite de lo que YOLO resuelve. Escalar el frame entero a mayor imgsz
# cuesta cuadratico (imgsz 1920 = 2.25x), pero la pelota esta SIEMPRE en una
# region chica y predecible. Recortando esa ventana a resolucion NATIVA se
# consigue la densidad de pixeles de un frame gigante al precio de una
# inferencia de 640 -- que es como lo resuelven los sistemas comerciales.
# --- HISTORIAL: el veto de "enganche a objeto quieto" fue PROBADO Y QUITADO ---
# Durante meses el detector parecia engancharse a objetos blancos e inmoviles
# (la marca de penal sobre todo). Se probaron CUATRO vetos, todos fallidos:
#   1. Geometrico: vetar sobre la marca Y sin moverse. El enganche EMPIEZA con
#      un salto desde la pelota real, y ese primer frame pasaba.
#   2. Geometrico por frames consecutivos sobre la marca (radio 1,2 m). Nunca
#      disparo: los enganches caen a 3,5-4,5 m de la marca NOMINAL.
#   3. Lista negra de celdas sobre-visitadas (`ball_unpin.py`). Bajo los frames
#      "clavados" 23,5%->3,8% pero la pelota cerca del penal quedo en 4,24%.
#   4. Veto por frame de TODA racha estatica. Clavados a 0,0% y la metrica del
#      sintoma SIN CAMBIO -- lo que rompio el caso.
# LA CAUSA ERA LA HOMOGRAFIA, no la pelota: la marca no estaba quieta en
# coordenadas de cancha porque la cancha se movia debajo de ella (ver
# MIN_HOMOGRAPHY_POINTS). Con la homografia arreglada, la pelota a <4 m de una
# marca de penal paso de 5,20% a 0,00% SIN ningun veto.
# El `StaticGuard` (soltar el ancla de continuidad tras N frames quieto) se
# implemento y se MIDIO sobre el clip ya arreglado: no aporta nada al sintoma
# (0,00% con y sin el) y ADEMAS mete teletransportes, porque al soltar el ancla
# el frame siguiente re-adquiere por confianza sin limite de distancia --
# movimientos imposibles 2,26% -> 4,26% y p99 de velocidad 37,8 -> 704 m/s en
# 103 sueltas. Por eso NO esta. No re-agregarlo sin volver a medir el p99.
#
# Margen (fraccion del campo) fuera del cual un candidato a pelota se descarta
# de entrada. La pelota sale del campo de verdad (lateral, corner), asi que el
# margen es generoso; lo que se corta son las botellas y las pelotas de
# calentamiento del costado. MEDIDO: 7,0% de los frames con pelota del clip
# caen fuera de los limites. Descartar en la SELECCION es estrictamente mejor
# que limpiar despues (clean_offpitch.py): aca todavia hay otro candidato al
# que recurrir, despues solo queda borrar el frame.
BALL_OFFPITCH_MARGIN = 0.04

BALL_CROP_SIZE = 640        # lado del recorte en pixeles nativos
BALL_CROP_CONF_SCALE = 1.0  # el recorte no cambia el umbral, solo la resolucion

# ByteTrack keeps a "lost" player track alive for this many frames before
# dropping it; a longer buffer reuses the same id across occlusions/missed
# detections instead of minting a fresh id on every reappearance (which
# fragmented players into thousands of short-lived ids).
DEFAULT_LOST_TRACK_BUFFER = 150

# Player tracks shorter than this many frames are treated as detector/tracker
# noise and discarded before event generation.
DEFAULT_MIN_TRACK_FRAMES = 12

# COCO class ids produced by the stock yolov8 weights.
# Equipo reservado para arqueros: ni Home (0) ni Away (1). Aguas abajo se los
# puede tratar aparte en vez de contar pases/perdidas contra un equipo inventado.
GOALKEEPER_TEAM_ID = 2

COCO_PERSON = 0
COCO_SPORTS_BALL = 32

# Standard pitch dimensions (centimetres) used to scale normalised image-space
# coordinates. Matches the dimensions assumed by ``lib.pitch`` for "raw" data.
PITCH_LENGTH_CM = 12000.0
PITCH_WIDTH_CM = 7000.0

# On frames with few/poor pitch keypoints the homography extrapolates wildly and
# sends detections far off the pitch. These margins (fractions of pitch size)
# bound that noise: the ball is dropped (and later interpolated) when beyond
# BALL, players are clamped into the PLAYER band.
BALL_PITCH_MARGIN = 0.10
PLAYER_PITCH_MARGIN = 0.15


def _pitch_in_bounds(xy, margin):
    return (-margin * PITCH_LENGTH_CM <= xy[0] <= (1 + margin) * PITCH_LENGTH_CM and
            -margin * PITCH_WIDTH_CM <= xy[1] <= (1 + margin) * PITCH_WIDTH_CM)


def _clamp_pitch(arr, margin):
    if len(arr) == 0:
        return arr
    arr = np.asarray(arr, dtype=float).copy()
    arr[:, 0] = np.clip(arr[:, 0], -margin * PITCH_LENGTH_CM, (1 + margin) * PITCH_LENGTH_CM)
    arr[:, 1] = np.clip(arr[:, 1], -margin * PITCH_WIDTH_CM, (1 + margin) * PITCH_WIDTH_CM)
    return arr


def image_to_pitch(detections, frame_w, frame_h):
    """Project detections to (approximate) pitch coordinates from image space.

    Uses each detection's bottom-centre pixel, normalised by the frame size and
    scaled to the standard pitch dimensions. No homography / keypoint model.
    """
    import supervision as sv

    if len(detections) == 0:
        return np.zeros((0, 2))
    anchors = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(float)
    pitch = np.empty_like(anchors)
    pitch[:, 0] = anchors[:, 0] / frame_w * PITCH_LENGTH_CM
    pitch[:, 1] = anchors[:, 1] / frame_h * PITCH_WIDTH_CM
    return pitch


def resolve_pitch_model_path(spec):
    """Resolve ``--pitch-model`` to a path, or None to disable homography."""
    if spec in (None, "none", "None", ""):
        return None
    if spec != "football-field":
        return spec
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=PITCH_MODEL_REPO, filename=PITCH_MODEL_FILE)
        print(f"Using pitch keypoint model {PITCH_MODEL_REPO} (homography enabled)")
        return path
    except Exception as exc:
        print(f"Could not fetch pitch model ({exc}); using image-space coords (no homography)")
        return None


def build_pitch_transformer(pitch_result, min_conf=DEFAULT_PITCH_CONF,
                            min_points=MIN_HOMOGRAPHY_POINTS,
                            frame_w=None, frame_h=None):
    """Build an image->pitch ``ViewTransformer`` from detected pitch keypoints.

    Devuelve ``(transformer, error_reproyeccion_cm, keypoints)`` o ``None`` si
    la solucion no pasa los controles de calidad (pocos puntos, puntos
    degenerados o error de reproyeccion alto), en cuyo caso el llamador reusa
    la ultima transformacion buena.
    """
    import supervision as sv
    try:
        from sports.configs.soccer import SoccerPitchConfiguration
        from sports.common.view import ViewTransformer
    except Exception:
        return None

    key_points = sv.KeyPoints.from_ultralytics(pitch_result)
    if key_points.xy is None or len(key_points.xy) == 0 or key_points.confidence is None:
        return None
    conf = key_points.confidence[0]
    mask = conf > min_conf
    if int(mask.sum()) < min_points:
        return None

    frame_pts = key_points.xy[0][mask].astype(np.float32)
    pitch_pts = np.array(SoccerPitchConfiguration().vertices)[mask].astype(np.float32)

    # Configuracion degenerada: puntos amontonados o casi alineados. Resuelven
    # una homografia exacta cerca de ellos y absurda lejos, y como el balon y
    # los jugadores suelen estar lejos, el disparate cae justo donde importa.
    span = frame_pts.max(axis=0) - frame_pts.min(axis=0)
    if frame_w and frame_h:
        if (span[0] < MIN_KEYPOINT_SPREAD * frame_w or
                span[1] < MIN_KEYPOINT_SPREAD * frame_h):
            return None
    centred = frame_pts - frame_pts.mean(axis=0)
    try:
        sing = np.linalg.svd(centred, compute_uv=False)
        if sing[0] <= 0 or sing[-1] / sing[0] < MIN_KEYPOINT_ASPECT:
            return None
    except np.linalg.LinAlgError:
        return None

    # Spread EN LA CANCHA. Sin esto se aceptan soluciones ancladas a un solo
    # rincon del campo (tipicamente un area de penal), que ajustan perfecto ahi
    # y proyectan a cientos de metros en el resto de la imagen.
    pspan = pitch_pts.max(axis=0) - pitch_pts.min(axis=0)
    if (pspan[0] < MIN_KEYPOINT_PITCH_X_CM or
            pspan[1] < MIN_KEYPOINT_PITCH_Y_CM):
        return None

    try:
        transformer = ViewTransformer(source=frame_pts, target=pitch_pts)
    except Exception:
        return None

    # Error de reproyeccion: con >= 6 puntos sobra informacion, asi que se
    # puede medir si la solucion es buena en vez de asumirlo. Con 4 puntos esto
    # daria 0 siempre y no diria nada.
    try:
        back = transformer.transform_points(points=frame_pts)
    except Exception:
        return None
    err = float(np.median(np.linalg.norm(back - pitch_pts, axis=1)))
    if not np.isfinite(err) or err > MAX_REPROJECTION_CM:
        return None
    return transformer, err, frame_pts


def _canonical_image_points(frame_w, frame_h):
    """Cuatro puntos fijos de la imagen, repartidos sobre la zona de juego.

    Sirven de "sonda": se mira a que parte de la cancha proyectan y se suaviza
    ESO. Se evitan los bordes porque cerca del horizonte la homografia es mal
    condicionada y amplifica cualquier ruido.
    """
    import numpy as np
    return np.array([
        [0.15 * frame_w, 0.35 * frame_h],
        [0.85 * frame_w, 0.35 * frame_h],
        [0.85 * frame_w, 0.90 * frame_h],
        [0.15 * frame_w, 0.90 * frame_h],
    ], dtype=np.float32)


def smooth_transformer(new_transformer, canon_img, prev_pitch,
                       alpha=HOMOGRAPHY_SMOOTH_ALPHA):
    """Promedio movil de la transformacion. Devuelve ``(transformer, pitch)``.

    ``prev_pitch`` es el estado suavizado anterior (o None la primera vez).
    """
    from sports.common.view import ViewTransformer

    new_pitch = new_transformer.transform_points(points=canon_img)
    if not np.all(np.isfinite(new_pitch)):
        return None, prev_pitch
    pitch = new_pitch if prev_pitch is None else \
        alpha * new_pitch + (1.0 - alpha) * prev_pitch
    try:
        return (ViewTransformer(source=canon_img,
                                target=pitch.astype(np.float32)), pitch)
    except Exception:
        return None, prev_pitch


def anchors_to_pitch(detections, transformer):
    """Project detection anchors to pitch coords via a homography transform."""
    import supervision as sv

    if len(detections) == 0:
        return np.zeros((0, 2))
    xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(np.float32)
    return transformer.transform_points(points=xy).astype(float)


def resolve_model_path(spec, fallback):
    """Resolve a model spec (``--player-model`` / ``--ball-model``) to a path.

    ``"football"`` downloads the football-trained YOLOv8 checkpoint from the
    Hugging Face Hub (no API key) and returns its local path, falling back to
    ``fallback`` if the download is unavailable. Any other value is returned
    unchanged (an explicit ``.pt`` path or an ultralytics model name).
    """
    if spec != "football":
        return spec
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=FOOTBALL_MODEL_REPO, filename=FOOTBALL_MODEL_FILE)
        print(f"Using football-specific weights {FOOTBALL_MODEL_REPO}/{FOOTBALL_MODEL_FILE}")
        return path
    except Exception as exc:  # network down, hub error, or missing huggingface_hub
        print(f"Could not fetch football model ({exc}); falling back to {fallback}")
        return fallback


def resolve_ball_class_id(model):
    """Find the model's "ball" class id so detection works for any weights.

    Football models label the ball as class 0; stock COCO weights use 32. We
    read the class id from the model's ``names`` map instead of hard-coding it.
    """
    names = getattr(model, "names", None) or {}
    for class_id, name in names.items():
        if str(name).strip().lower() in ("ball", "sports ball", "soccer ball", "football"):
            return int(class_id)
    return COCO_SPORTS_BALL


def resolve_goalkeeper_class_id(model):
    """Class id del arquero, o None si el modelo no lo distingue.

    El clasificador de equipos hace k-means con DOS grupos sobre el color de
    camiseta, y el arquero usa un tercer color: cae en cualquiera de los dos al
    azar. Medido en spain-france: los arqueros salieron 58/42 y 62/38 entre
    Home y Away, o sea sin senal. Conservar la clase del detector es la unica
    forma limpia de saber quien es arquero; el color no alcanza.
    """
    names = getattr(model, "names", None) or {}
    for class_id, name in names.items():
        if str(name).strip().lower() in ("goalkeeper", "goal keeper", "gk"):
            return int(class_id)
    return None


def resolve_player_class_ids(model):
    """Class ids the tracker should treat as players (players + goalkeepers).

    Football models expose ball/goalkeeper/player/referee; we keep players and
    goalkeepers and drop the ball and the referee (so referees don't pollute the
    player tracks or the team classifier). Stock COCO weights only have "person"
    (id 0), which is then used as the fallback.
    """
    names = getattr(model, "names", None) or {}
    ids = [int(class_id) for class_id, name in names.items()
           if str(name).strip().lower() in ("player", "goalkeeper", "goal keeper", "person")]
    return ids or [COCO_PERSON]


def _ball_crop_window(last_px, vel_px, frame_w, frame_h, size=BALL_CROP_SIZE):
    """Ventana (x0, y0, x1, y1) centrada donde se PREDICE que esta la pelota.

    Se proyecta la ultima posicion con su velocidad: en una pelota rapida el
    centro del recorte tiene que adelantarse, si no la jugada se escapa de la
    ventana justo cuando mas importa (un pase largo).
    """
    cx = last_px[0] + vel_px[0]
    cy = last_px[1] + vel_px[1]
    half = size // 2
    x0 = int(max(0, min(frame_w - size, cx - half)))
    y0 = int(max(0, min(frame_h - size, cy - half)))
    return x0, y0, x0 + size, y0 + size


def _ball_offpitch(x, y, margin=BALL_OFFPITCH_MARGIN):
    """True si un candidato a pelota cae fuera del campo con margen."""
    return (x < -margin * PITCH_LENGTH_CM or x > (1 + margin) * PITCH_LENGTH_CM or
            y < -margin * PITCH_WIDTH_CM or y > (1 + margin) * PITCH_WIDTH_CM)


def _pick_ball(ball_detections, ball_pitch, last_pitch, gap_frames,
               fps=DEFAULT_TRACK_FPS):
    """Choose one ball detection, preferring trajectory continuity.

    Works in PITCH COORDINATES (centimetres), never pixels. An earlier version
    gated in pixel space and only cut impossible moves 24.5% -> 12.1%, because
    the homography is perspective: the same pixel distance is a few metres near
    the camera and many metres at the far touchline, so a pixel radius is far
    too permissive exactly where play is furthest away. Distances are compared
    in metres, which is also the unit the "impossible move" metric uses.

    ``last_pitch`` is the previously accepted ball position (cm) or None, and
    ``gap_frames`` how many frames ago it was seen. Reachability is a HARD
    filter -- confidence only breaks ties among physically possible candidates,
    so a high-confidence pitch marking cannot outvote a faint real ball.
    Returns empty selections when nothing is reachable: leaving the frame blank
    (interpolate_ball bridges it) beats recording an impossible position.

    Always index Detections with a SLICE (``[i:i+1]``): supervision validates
    that ``xyxy`` stays 2-D (N, 4) and scalar indexing raises deep inside
    ``Detections.__post_init__``.
    """
    conf = ball_detections.confidence
    if len(ball_detections) == 0:
        return ball_detections, ball_pitch

    # Reachable radius in centimetres, growing with the blank gap so a
    # genuinely lost ball is re-acquired instead of suppressed forever.
    radius_cm = (MAX_BALL_SPEED_MS * max(1, gap_frames) / max(1.0, fps)) * 100.0

    ok = []
    for i, (x, y) in enumerate(ball_pitch):
        # Fuera del campo: descarte DURO. Una botella en la banda no es la
        # pelota por mucha confianza que tenga el detector.
        if _ball_offpitch(x, y):
            continue
        if last_pitch is None:
            dist = 0.0
        else:
            dist = ((x - last_pitch[0]) ** 2 + (y - last_pitch[1]) ** 2) ** 0.5
            if dist > radius_cm:
                continue
        c = float(conf[i]) if conf is not None else 1.0
        ok.append((c, -dist, i))

    if not ok:
        return ball_detections[0:0], ball_pitch[0:0]

    best = max(ok)[2]
    return ball_detections[best:best + 1], ball_pitch[best:best + 1]


def get_detections(frame, player_result, ball_result, tracker, team_classifier,
                   frame_w, frame_h, ball_class_id=COCO_SPORTS_BALL,
                   player_class_ids=(COCO_PERSON,), player_conf=DEFAULT_PLAYER_CONF,
                   goalkeeper_class_id=None,
                   transformer=None, last_ball_pitch=None, frames_since_ball=1,
                   ball_crop_result=None, crop_offset=(0, 0),
                   effective_fps=DEFAULT_TRACK_FPS):
    import supervision as sv

    # Players: keep the player/goalkeeper classes (drop ball & referee). When a
    # single shared model is run at the low ball confidence, also drop player
    # boxes below ``player_conf`` so player behaviour stays unchanged.
    detections = sv.Detections.from_ultralytics(player_result)
    players_detections = detections[np.isin(detections.class_id, list(player_class_ids))]
    if players_detections.confidence is not None:
        players_detections = players_detections[players_detections.confidence >= player_conf]
    players_detections = players_detections.with_nms(threshold=0.5, class_agnostic=True)
    players_detections = tracker.update_with_detections(detections=players_detections)
    # Se lee DESPUES del filtro de confianza, el NMS y el tracker (que cambian
    # cuantas detecciones hay) y ANTES del clasificador de equipos, que es
    # quien sobrescribe ``class_id`` con el equipo.
    gk_mask = (players_detections.class_id == goalkeeper_class_id
               if goalkeeper_class_id is not None
               and players_detections.class_id is not None else None)

    if team_classifier and len(players_detections):
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        teams = np.asarray(team_classifier.predict(players_crops))
        # El arquero lleva un tercer color de camiseta y el clasificador solo
        # tiene DOS grupos, asi que lo asigna al azar (medido: 58/42 y 62/38).
        # Se lo marca como equipo 2 en vez de forzarlo a uno de los dos: mejor
        # "no se" que un dato inventado que despues genera pases y perdidas
        # fantasma contra el arquero.
        if goalkeeper_class_id is not None and gk_mask is not None and len(gk_mask):
            teams = np.where(gk_mask, GOALKEEPER_TEAM_ID, teams)
        players_detections.class_id = teams
    else:
        players_detections.class_id = np.zeros(len(players_detections), dtype=int)

    # Ball: filter to the model's ball class (0 for football models, 32 for COCO).
    ball_all = sv.Detections.from_ultralytics(ball_result)
    ball_detections = ball_all[ball_all.class_id == ball_class_id]

    # Candidatos extra del recorte de alta resolucion. Sus coordenadas son
    # relativas al recorte, asi que se trasladan al frame completo antes de
    # mezclarlos: a partir de aca son un candidato mas, indistinguible de los
    # del frame entero, y el gate de continuidad decide.
    if ball_crop_result is not None:
        crop_all = sv.Detections.from_ultralytics(ball_crop_result)
        crop_ball = crop_all[crop_all.class_id == ball_class_id]
        if len(crop_ball):
            crop_ball.xyxy = crop_ball.xyxy + np.array(
                [crop_offset[0], crop_offset[1], crop_offset[0], crop_offset[1]],
                dtype=crop_ball.xyxy.dtype)
            ball_detections = sv.Detections.merge([ball_detections, crop_ball]) \
                if len(ball_detections) else crop_ball
    # The ball is unique, so only one box survives. Picking purely by
    # confidence is WRONG on a fixed camera: the penalty spot, pitch markings
    # and crowd objects are small white blobs that regularly out-score the real
    # ball, and each time they do the stored position teleports. Measured on
    # spain-france: 25% of consecutive ball movements were physically
    # impossible (p90 = 417 m/s = 1502 km/h), which is what makes possession
    # flicker and defeats any physics-based possession logic downstream.
    # So candidates are scored by CONTINUITY first: a candidate reachable from
    # the last known position at a plausible speed beats a more confident one
    # that would require teleporting.

    # Project to pitch coords. Image-space is always on-pitch but approximate;
    # the homography is accurate but goes wild on bad-keypoint frames. So: keep
    # the (approximate) image-space projection as a safety net and only use the
    # homography when it's trustworthy for this frame -- never dropping the ball.
    img_players = image_to_pitch(players_detections, frame_w, frame_h)
    img_ball = image_to_pitch(ball_detections, frame_w, frame_h)

    if transformer is None:
        players_pitch, ball_pitch = img_players, img_ball
    else:
        h_players = anchors_to_pitch(players_detections, transformer)
        h_ball = anchors_to_pitch(ball_detections, transformer)
        # If many players land off the pitch, the homography is unreliable this
        # frame -> fall back to image-space for everything (keeps it consistent).
        off_frac = (float(np.mean([not _pitch_in_bounds(p, PLAYER_PITCH_MARGIN)
                                   for p in h_players])) if len(h_players) else 0.0)
        if off_frac > 0.3:
            players_pitch, ball_pitch = img_players, img_ball
        else:
            players_pitch = _clamp_pitch(h_players, PLAYER_PITCH_MARGIN)
            # Keep the ball; if its homography point is wild, use image-space
            # (approximate) instead of discarding the detection.
            if len(h_ball) and not _pitch_in_bounds(h_ball[0], BALL_PITCH_MARGIN):
                ball_pitch = img_ball
            else:
                ball_pitch = h_ball

    # Snapshot of EVERY ball candidate with its pitch position, before the
    # greedy gate throws all but one away. The gate below is causal (it only
    # looks backwards), so it can lock onto a wrong object and then reject the
    # real ball as "unreachable" for a while. Keeping the candidates lets
    # ball_viterbi.py re-solve the whole trajectory globally afterwards, which
    # recovers exactly those cases -- without re-running detection.
    ball_pitch_all = np.asarray(ball_pitch, dtype=float).reshape(-1, 2)
    candidates = []
    if len(ball_detections):
        conf_all = ball_detections.confidence
        order = (np.argsort(-conf_all)[:BALL_CANDIDATES_PER_FRAME]
                 if conf_all is not None
                 else range(min(BALL_CANDIDATES_PER_FRAME, len(ball_detections))))
        for i in order:
            box = ball_detections.xyxy[i]
            candidates.append((
                float(conf_all[i]) if conf_all is not None else 1.0,
                float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                float(ball_pitch_all[i][0]), float(ball_pitch_all[i][1]),
            ))

    # Continuity gate LAST, now that ball candidates have real pitch
    # coordinates: the physics of "how far can a ball travel" only makes sense
    # in metres, not pixels (see _pick_ball).
    ball_detections, ball_pitch = _pick_ball(
        ball_detections, ball_pitch_all,
        last_ball_pitch, frames_since_ball, fps=effective_fps)

    players_detections.data["pitch_xy"] = players_pitch
    ball_detections.data["pitch_xy"] = ball_pitch

    return players_detections, ball_detections, candidates


def generate_team_model(video_path, player_model, player_class_ids=(COCO_PERSON,),
                        stride=30, imgsz=DEFAULT_IMGSZ):
    import supervision as sv
    from sports.common.team import TeamClassifier

    frame_generator = sv.get_video_frames_generator(source_path=video_path, stride=stride)
    crops = []
    for frame in tqdm(frame_generator, desc="collecting crops"):
        result = player_model(frame, conf=DEFAULT_PLAYER_CONF, imgsz=imgsz, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[np.isin(detections.class_id, list(player_class_ids))]
        crops += [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

    team_classifier = TeamClassifier(device="cuda") if _cuda_available() else TeamClassifier()
    team_classifier.fit(crops)
    return team_classifier


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _ball_is_empty(entry):
    """A stored ball entry is "empty" when no detection was found that frame."""
    return (entry is None or
            (entry["X1"] == 0 and entry["Y1"] == 0 and
             entry["X2"] == 0 and entry["Y2"] == 0))


def interpolate_ball(ball, total_frames, max_gap=BALL_INTERP_MAX_GAP):
    """Linearly interpolate the ball across short detection gaps, in place.

    A "gap" is a run of consecutive frames with no ball detection bounded on
    both sides by a real detection. Gaps no longer than ``max_gap`` frames are
    filled by linearly interpolating every stored coordinate (bounding box +
    pitch position) between the last and next known frames; longer gaps, and
    gaps at the very start/end of the clip, are left empty. This keeps the ball
    "present" through brief misses so possession isn't reset downstream.

    Returns the number of frames filled.
    """
    keys = ["X1", "Y1", "X2", "Y2", "X_Pitch", "Y_Pitch"]
    filled = 0
    frame = 1
    while frame <= total_frames:
        start = ball.get(str(frame))
        if _ball_is_empty(start):
            frame += 1
            continue
        # Walk forward to the next detected frame.
        nxt = frame + 1
        while nxt <= total_frames and _ball_is_empty(ball.get(str(nxt))):
            nxt += 1
        gap = nxt - frame - 1  # number of missing frames between the two detections
        if nxt <= total_frames and 0 < gap <= max_gap:
            end = ball[str(nxt)]
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                ball[str(frame + k)] = {
                    key: start[key] + (end[key] - start[key]) * t for key in keys
                }
            filled += gap
        frame = nxt if nxt > frame else frame + 1
    return filled


def drop_short_tracks(players, min_frames):
    """Remove player tracks present in fewer than ``min_frames`` frames.

    Returns ``(kept_players, dropped_count)``. These ultra-short tracks are the
    fragments ByteTrack mints when it loses and re-IDs a player; dropping them
    cuts phantom turnovers in the events without touching real players.
    """
    if min_frames <= 1:
        return players, 0
    kept = {pid: data for pid, data in players.items() if len(data) >= min_frames}
    return kept, len(players) - len(kept)


def save_tracking_results(players, ball, frames, output_path):
    # Bucket player rows by frame in a single pass, then emit in frame order.
    # This is linear in the number of rows; the previous ``csv += ...`` in a
    # double loop was O(frames x players) with O(n^2) string growth, which
    # stalls on full 90-minute matches (~millions of rows).
    by_frame = {}
    for player_id, player_data in players.items():
        for f_str, d in player_data.items():
            by_frame.setdefault(int(f_str), []).append(",".join(map(str, [
                int(f_str), "player", player_id, d["Team"],
                d["X1"], d["Y1"], d["X2"], d["Y2"], d["X_Pitch"], d["Y_Pitch"],
            ])))

    rows = ["Frame,Object,Object ID,Team,X1,Y1,X2,Y2,X_Pitch,Y_Pitch"]
    for frame in range(1, frames):
        rows.extend(by_frame.get(frame, []))
        if str(frame) in ball:
            d = ball[str(frame)]
            rows.append(",".join(map(str, [
                frame, "ball", "", "",
                d["X1"], d["Y1"], d["X2"], d["Y2"], d["X_Pitch"], d["Y_Pitch"],
            ])))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(rows) + "\n")
    return output_path


def track(video_path, output_dir,
          player_model_path=DEFAULT_PLAYER_MODEL,
          ball_model_path=DEFAULT_BALL_MODEL,
          ball_conf=DEFAULT_BALL_CONF, ball_interp_gap=BALL_INTERP_MAX_GAP,
          track_buffer=DEFAULT_LOST_TRACK_BUFFER,
          min_track_frames=DEFAULT_MIN_TRACK_FRAMES, imgsz=DEFAULT_IMGSZ,
          pitch_model_path=DEFAULT_PITCH_MODEL, pitch_conf=DEFAULT_PITCH_CONF,
          homography_every=DEFAULT_HOMOGRAPHY_EVERY, pitch_imgsz=DEFAULT_PITCH_IMGSZ,
          track_teams=True, generate_video=False, stride=30, frame_stride=1,
          ball_crop=False):
    import supervision as sv
    from ultralytics import YOLO

    # Resolve and load the player model (football weights by default).
    resolved_player_path = resolve_model_path(player_model_path, FALLBACK_PLAYER_MODEL)
    try:
        player_model = YOLO(resolved_player_path)
    except Exception as exc:
        print(f"Failed to load player model '{resolved_player_path}' ({exc}); "
              f"using {FALLBACK_PLAYER_MODEL}")
        resolved_player_path = FALLBACK_PLAYER_MODEL
        player_model = YOLO(FALLBACK_PLAYER_MODEL)
    player_class_ids = resolve_player_class_ids(player_model)
    goalkeeper_class_id = resolve_goalkeeper_class_id(player_model)
    if goalkeeper_class_id is not None:
        print(f"Arqueros identificados por clase (id {goalkeeper_class_id}) y "
              f"marcados como equipo {GOALKEEPER_TEAM_ID} (el clasificador de "
              f"color no los distingue).")

    # Resolve the ball model. If it resolves to the same weights as the player
    # model, reuse it and run a single inference per frame.
    resolved_ball_path = resolve_model_path(ball_model_path, FALLBACK_BALL_MODEL)
    share_model = (resolved_ball_path == resolved_player_path)
    if share_model:
        ball_model = player_model
        print("Player and ball share one model: running a single inference per frame.")
    else:
        try:
            ball_model = YOLO(resolved_ball_path)
        except Exception as exc:
            print(f"Failed to load ball model '{resolved_ball_path}' ({exc}); "
                  f"using {FALLBACK_BALL_MODEL}")
            ball_model = YOLO(FALLBACK_BALL_MODEL)
    ball_class_id = resolve_ball_class_id(ball_model)

    # Pitch keypoint model for homography (optional; falls back to image space).
    pitch_model = None
    resolved_pitch_path = resolve_pitch_model_path(pitch_model_path)
    if resolved_pitch_path:
        try:
            pitch_model = YOLO(resolved_pitch_path)
        except Exception as exc:
            print(f"Failed to load pitch model '{resolved_pitch_path}' ({exc}); "
                  f"using image-space coords (no homography)")
            pitch_model = None

    ellipse_annotator = triangle_annotator = label_annotator = None
    if generate_video:
        ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']), thickness=2)
        label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000'), text_position=sv.Position.BOTTOM_CENTER)
        triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'), base=25, height=21, outline_thickness=1)

    video_info = sv.VideoInfo.from_video_path(video_path=video_path)
    frame_w, frame_h = video_info.width, video_info.height
    fps = int(round(video_info.fps)) or 24

    # ``frame_stride`` runs detection on every Nth video frame: N=2 halves the
    # GPU cost. The written frame numbers stay CONSECUTIVE (1..M over processed
    # frames) because lib/ball.py indexes ``frames[frame_number - 1]`` and
    # event_generator walks ``range(1, match.frames + 1)`` -- gaps would break
    # both. The real timeline is preserved via the sidecar ``effective_fps``.
    frame_stride = max(1, int(frame_stride))
    effective_fps = fps / frame_stride

    # Every frame-count parameter below is expressed in *processed* frames, so
    # dividing by the stride keeps its meaning in SECONDS unchanged.
    if frame_stride > 1:
        track_buffer = max(1, round(track_buffer / frame_stride))
        min_track_frames = max(1, round(min_track_frames / frame_stride))
        ball_interp_gap = max(1, round(ball_interp_gap / frame_stride))
        # ``homography_every`` is deliberately NOT rescaled: it stays in
        # processed frames, so the pitch model runs ``frame_stride`` times less
        # often in wall-clock terms. On a fixed/tactical camera the homography
        # barely moves, so that costs nothing and pays for the second detection
        # pass. Lower it manually if the camera pans a lot.
        print(f"frame-stride {frame_stride}: procesando 1 de cada {frame_stride} frames "
              f"({fps} fps -> {effective_fps:g} fps efectivos).")
        print(f"  reescalados (conservan su valor en segundos): "
              f"track-buffer={track_buffer}, min-track-frames={min_track_frames}, "
              f"ball-interp-gap={ball_interp_gap}")
        print(f"  homography-every={homography_every} sin reescalar "
              f"(refresca cada {homography_every * frame_stride} frames de video; "
              f"OK con camara fija)")

    # Longer lost-track buffer => fewer fragmented ids across occlusions.
    try:
        tracker = sv.ByteTrack(lost_track_buffer=track_buffer,
                               frame_rate=max(1, int(round(effective_fps))))
    except TypeError:  # older supervision signature without these kwargs
        tracker = sv.ByteTrack()
    tracker.reset()

    team_classifier = None
    if track_teams:
        try:
            team_classifier = generate_team_model(
                video_path, player_model, player_class_ids=player_class_ids,
                stride=stride, imgsz=imgsz)
        except Exception as exc:
            # The team classifier pulls a heavy umap/numba stack that can clash
            # with the runtime's NumPy. Degrade gracefully instead of crashing:
            # tracking still runs, players just aren't split into teams.
            print(f"⚠️  Team classifier unavailable ({exc}); continuing WITHOUT "
                  f"team labels (events will be less detailed).")
            team_classifier = None

    frame_generator = sv.get_video_frames_generator(video_path)

    players, ball = {}, {}
    ball_candidates = []      # (frame, conf, x1, y1, x2, y2, x_pitch, y_pitch)
    frame_number = 1
    # Last accepted ball position (pitch cm) + how long ago, for _pick_ball.
    last_ball_pitch, frames_since_ball = None, 1
    # Posicion y velocidad de la pelota en PIXELES, para centrar el recorte.
    last_ball_px, ball_vel_px = None, (0.0, 0.0)
    transformer = None  # most recent image->pitch homography
    last_homog_err = 1e9        # error de reproyeccion de la transformacion vigente
    n_homog_ok = n_homog_rejected = n_homog_jumps = 0
    canon_img = _canonical_image_points(frame_w, frame_h)
    smooth_pitch = None         # estado del promedio movil de la homografia

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    annotated_video_path = os.path.join(output_dir, video_name + "_tracked.mp4")
    sink = sv.VideoSink(target_path=annotated_video_path, video_info=video_info) if generate_video else None
    if sink:
        sink.__enter__()

    shared_conf = min(DEFAULT_PLAYER_CONF, ball_conf)
    for source_index, frame in enumerate(tqdm(frame_generator, desc="Collecting Tracking Data...")):
        # Skip the frames the stride drops *before* any inference -- decoding is
        # cheap, the two YOLO passes are what costs.
        if source_index % frame_stride:
            continue

        if share_model:
            # One pass at the lower confidence; players are re-thresholded in
            # get_detections so their behaviour is unchanged.
            result = player_model(frame, conf=shared_conf, imgsz=imgsz, verbose=False)[0]
            player_result = ball_result = result
        else:
            player_result = player_model(frame, conf=DEFAULT_PLAYER_CONF, imgsz=imgsz, verbose=False)[0]
            ball_result = ball_model(frame, conf=ball_conf, imgsz=imgsz, verbose=False)[0]

        # Recorte de alta resolucion alrededor de la pelota predicha. Solo se
        # hace si ya sabemos donde estaba: sin historial no hay donde recortar.
        ball_crop_result, crop_offset = None, (0, 0)
        if ball_crop and last_ball_px is not None:
            x0, y0, x1, y1 = _ball_crop_window(
                last_ball_px, ball_vel_px, frame_w, frame_h)
            crop = frame[y0:y1, x0:x1]
            if crop.size:
                # imgsz = el lado del recorte: se procesa a resolucion NATIVA,
                # sin reescalar, que es todo el truco.
                ball_crop_result = ball_model(crop, conf=ball_conf,
                                              imgsz=BALL_CROP_SIZE,
                                              verbose=False)[0]
                crop_offset = (x0, y0)

        # Refresh the homography every ``homography_every`` frames; reuse the
        # last good transform in between (and keep it if a frame has too few
        # keypoints to solve a new one).
        if pitch_model is not None and (frame_number - 1) % homography_every == 0:
            pitch_result = pitch_model(frame, imgsz=min(pitch_imgsz, imgsz), verbose=False)[0]
            built = build_pitch_transformer(pitch_result, min_conf=pitch_conf,
                                            frame_w=frame_w, frame_h=frame_h)
            if built is None:
                n_homog_rejected += 1
            else:
                new_transformer, new_err, kp = built
                accept = True
                if transformer is not None:
                    # CONTINUIDAD: se proyectan los MISMOS pixeles con la
                    # transformacion vieja y la nueva. Si el mundo se corre
                    # varios metros de un refresco al otro, una de las dos esta
                    # rota; se conserva la vieja salvo que la nueva reproyecte
                    # claramente mejor (asi se puede salir de una mala, en vez
                    # de quedar clavado en ella para siempre).
                    try:
                        old_xy = transformer.transform_points(points=kp)
                        jump = float(np.median(
                            np.linalg.norm(old_xy - new_transformer.transform_points(points=kp),
                                           axis=1)))
                    except Exception:
                        jump = 0.0
                    if jump > MAX_HOMOGRAPHY_JUMP_CM and new_err >= last_homog_err * 0.5:
                        accept = False
                        n_homog_jumps += 1
                if accept:
                    # Suavizado temporal: corta el ruido de alta frecuencia de
                    # los keypoints SIN congelar el mapa (que es lo que arruinó
                    # la calibración en la version anterior).
                    sm, smooth_pitch = smooth_transformer(
                        new_transformer, canon_img, smooth_pitch)
                    transformer = sm if sm is not None else new_transformer
                    last_homog_err = max(new_err, 1.0)
                    n_homog_ok += 1

        all_detections, ball_detections, ball_cands = get_detections(
            frame, player_result, ball_result, tracker, team_classifier, frame_w, frame_h,
            ball_class_id=ball_class_id, player_class_ids=player_class_ids,
            goalkeeper_class_id=goalkeeper_class_id,
            player_conf=DEFAULT_PLAYER_CONF, transformer=transformer,
            last_ball_pitch=last_ball_pitch, frames_since_ball=frames_since_ball,
            effective_fps=effective_fps,
            ball_crop_result=ball_crop_result, crop_offset=crop_offset)

        for cand in ball_cands:
            ball_candidates.append((frame_number,) + cand)

        object_ids = all_detections.tracker_id
        team_ids = all_detections.class_id
        pitch_xys = all_detections.data["pitch_xy"]
        ball_pitch_xys = ball_detections.data["pitch_xy"]
        all_detections.class_id = all_detections.class_id.astype(int)

        for idx, xyxy in enumerate(all_detections.xyxy):
            object_id = str(object_ids[idx])
            players.setdefault(object_id, {})
            players[object_id][str(frame_number)] = {
                "Team": team_ids[idx],
                "X1": xyxy[0], "Y1": xyxy[1], "X2": xyxy[2], "Y2": xyxy[3],
                "X_Pitch": pitch_xys[idx][0], "Y_Pitch": pitch_xys[idx][1],
            }

        if ball_detections.xyxy.shape[0]:
            b = ball_detections
            last_ball_pitch = (ball_pitch_xys[0][0], ball_pitch_xys[0][1])
            _bb = b.xyxy[0]
            px = ((_bb[0] + _bb[2]) / 2.0, (_bb[1] + _bb[3]) / 2.0)
            if last_ball_px is not None:
                ball_vel_px = (px[0] - last_ball_px[0], px[1] - last_ball_px[1])
            last_ball_px = px
            frames_since_ball = 1
            ball[str(frame_number)] = {
                "X1": b.xyxy[0][0], "Y1": b.xyxy[0][1], "X2": b.xyxy[0][2], "Y2": b.xyxy[0][3],
                "X_Pitch": ball_pitch_xys[0][0], "Y_Pitch": ball_pitch_xys[0][1],
            }
        else:
            frames_since_ball += 1
            ball[str(frame_number)] = {
                "X1": 0, "Y1": 0, "X2": 0, "Y2": 0, "X_Pitch": 0, "Y_Pitch": 0,
            }

        if generate_video:
            labels = [f"#{tid}" for tid in all_detections.tracker_id]
            annotated = frame.copy()
            annotated = ellipse_annotator.annotate(scene=annotated, detections=all_detections)
            annotated = label_annotator.annotate(scene=annotated, detections=all_detections, labels=labels)
            annotated = triangle_annotator.annotate(scene=annotated, detections=ball_detections)
            sink.write_frame(frame=annotated)

        frame_number += 1

    if sink:
        sink.__exit__(None, None, None)

    total_h = n_homog_ok + n_homog_rejected + n_homog_jumps
    if total_h:
        print(f"Homografia: {n_homog_ok} aceptadas, {n_homog_rejected} descartadas "
              f"por calidad, {n_homog_jumps} descartadas por salto "
              f"({100.0 * (n_homog_rejected + n_homog_jumps) / total_h:.1f}% rechazo)")

    # Drop fragmented (ultra-short) player tracks before events.
    players, dropped = drop_short_tracks(players, min_track_frames)
    if dropped:
        print(f"Dropped {dropped} short player track(s) (< {min_track_frames} frames); "
              f"{len(players)} tracks kept.")

    # Bridge short ball-detection gaps so possession survives a few missed frames.
    filled = interpolate_ball(ball, frame_number - 1, max_gap=ball_interp_gap)
    if filled:
        print(f"Interpolated the ball across {filled} missed frame(s) "
              f"(gaps <= {ball_interp_gap}).")

    cand_path = os.path.join(output_dir, video_name + "_ball_candidates.csv")
    with open(cand_path, "w", newline="") as fh:
        fh.write("Frame,Conf,X1,Y1,X2,Y2,X_Pitch,Y_Pitch\n")
        fh.writelines(
            f"{c[0]},{c[1]:.4f},{c[2]:.1f},{c[3]:.1f},{c[4]:.1f},{c[5]:.1f},"
            f"{c[6]:.1f},{c[7]:.1f}\n" for c in ball_candidates)
    print(f"Candidatos de pelota: {cand_path} ({len(ball_candidates)} filas)")

    output_path = os.path.join(output_dir, video_name + ".csv")
    save_tracking_results(players, ball, frame_number, output_path)

    # Sidecar so downstream tools recover the real timeline: with a stride the
    # CSV's consecutive frame numbers tick at ``effective_fps``, not the video's
    # fps, and every timestamp would be off by the stride factor without this.
    meta_path = os.path.join(output_dir, video_name + ".meta.json")
    with open(meta_path, "w") as meta_file:
        json.dump({
            "video": os.path.basename(video_path),
            "source_fps": fps,
            "frame_stride": frame_stride,
            "effective_fps": effective_fps,
            "tracked_frames": frame_number - 1,
        }, meta_file, indent=2)
    print(f"Metadata: {meta_path} (effective_fps={effective_fps:g})")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 (ultralytics) tracking on match footage.")
    parser.add_argument("--video", required=True, help="Path to the input .mp4 footage.")
    parser.add_argument("--output", default="./track/output", help="Directory for the tracking CSV.")
    parser.add_argument("--player-model", default=DEFAULT_PLAYER_MODEL,
                        help="Weights for player detection. 'football' (default) downloads a "
                             "football-trained model from the HF Hub (fallback yolov8n.pt); "
                             "or pass a path to a .pt file.")
    parser.add_argument("--ball-model", default=DEFAULT_BALL_MODEL,
                        help="Weights for ball detection. 'football' (default) downloads a "
                             "football-trained model from the HF Hub (fallback yolov8m.pt); "
                             "or pass a path to a .pt file.")
    parser.add_argument("--ball-conf", type=float, default=DEFAULT_BALL_CONF,
                        help="Confidence threshold for ball detection (default 0.15; the ball "
                             "is small and easily missed).")
    parser.add_argument("--ball-interp-gap", type=int, default=BALL_INTERP_MAX_GAP,
                        help="Max consecutive missed frames to linearly interpolate the ball "
                             "across (default 15).")
    parser.add_argument("--track-buffer", type=int, default=DEFAULT_LOST_TRACK_BUFFER,
                        help="ByteTrack lost-track buffer in frames: how long a lost player "
                             "keeps its id before a new one is minted (default 90).")
    parser.add_argument("--min-track-frames", type=int, default=DEFAULT_MIN_TRACK_FRAMES,
                        help="Discard player tracks shorter than this many frames (default 12).")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help="Inference resolution (default 1280). Higher recovers more small "
                             "ball detections but is slower; 640 is the fast/low-recall option.")
    parser.add_argument("--pitch-model", default=DEFAULT_PITCH_MODEL,
                        help="Pitch keypoint model for homography. 'football-field' (default) "
                             "downloads it from the HF Hub; 'none' disables homography and uses "
                             "image-space coords; or pass a path to a .pt file.")
    parser.add_argument("--pitch-conf", type=float, default=DEFAULT_PITCH_CONF,
                        help="Min keypoint confidence used in the homography (default 0.5).")
    parser.add_argument("--homography-every", type=int, default=DEFAULT_HOMOGRAPHY_EVERY,
                        help="Recompute the homography every N frames (default 5).")
    parser.add_argument("--pitch-imgsz", type=int, default=DEFAULT_PITCH_IMGSZ,
                        help="Resolution cap for the pitch keypoint model (default 1280); kept "
                             "lower than --imgsz since field landmarks don't need high res.")
    parser.add_argument("--no-teams", action="store_true", help="Disable team classification.")
    parser.add_argument("--generate-video", action="store_true", help="Also write an annotated video.")
    parser.add_argument("--stride", type=int, default=30, help="Frame stride for team-model crops.")
    parser.add_argument("--ball-crop", action="store_true",
                        help="Ademas del frame completo, corre el modelo de "
                             "pelota sobre un recorte de 640 px a resolucion "
                             "NATIVA centrado donde se predice la pelota. A "
                             "1080p la pelota mide ~9 px; en el recorte "
                             "conserva su tamano real. Cuesta como una "
                             "inferencia de 640 (barato) y da candidatos que "
                             "el frame completo pierde.")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Run detection on every Nth video frame (2 = half the GPU cost). "
                             "Frame-count options are rescaled automatically so their meaning "
                             "in seconds is unchanged; the true timeline is recorded in the "
                             "sidecar .meta.json. Higher values fragment tracks more.")
    return parser.parse_args()


def main():
    args = parse_args()
    out = track(
        video_path=args.video,
        output_dir=args.output,
        player_model_path=args.player_model,
        ball_model_path=args.ball_model,
        ball_conf=args.ball_conf,
        ball_interp_gap=args.ball_interp_gap,
        track_buffer=args.track_buffer,
        min_track_frames=args.min_track_frames,
        imgsz=args.imgsz,
        pitch_model_path=args.pitch_model,
        pitch_conf=args.pitch_conf,
        homography_every=args.homography_every,
        pitch_imgsz=args.pitch_imgsz,
        track_teams=not args.no_teams,
        generate_video=args.generate_video,
        stride=args.stride,
        frame_stride=args.frame_stride,
        ball_crop=args.ball_crop,
    )
    print(f"Tracking data written to: {out}")


if __name__ == "__main__":
    main()
