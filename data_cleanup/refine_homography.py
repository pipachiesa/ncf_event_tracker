"""Refina una homografia alineando las lineas proyectadas con las PINTADAS.

LA IDEA
El modelo de keypoints pone los vertices ~4,4 m fuera de lugar (medido contra
el ground truth del frame 761, con confianzas de 0,94-0,99). Ninguna forma de
combinar esos puntos arregla eso: hay que corregirlos contra algo que este
siempre en la imagen y no dependa del modelo. Las lineas blancas lo estan.

Se construye una mascara de lineas (brillante, poco saturada, mas clara que sus
vecinas a izquierda y derecha, y DENTRO del cesped) y se optimizan los 8 grados
de libertad de la homografia para minimizar la distancia media entre las lineas
del modelo de cancha proyectadas y la linea blanca mas cercana.

POR QUE GENERALIZA
No usa anotacion manual ni supone una camara concreta: las lineas de una cancha
son las mismas en cualquier estadio, y la mascara se calcula por frame. El
modelo de keypoints queda como INICIALIZACION, que es para lo que sirve.

MEDIDO sobre el frame 761 del clip, partiendo del ground truth a mano:

    distancia media a la linea blanca mas cercana:  12,7 px -> 3,9 px
    los landmarks se movieron 10-14 px

Esos 10-14 px son los que Felipe vio a ojo cuando le mostre el ground truth
("estan alineadas pero un poco corridas"). En el campo lejano son metros.

Lo que NO mejora son los elementos ocluidos (el area chica detras del arquero,
el lateral cercano donde esta el banco de suplentes): ahi el limite es la
mascara, no el ajuste.
"""

import numpy as np

# Cuatro puntos de CANCHA repartidos por la zona que suele verse. Se optimizan
# sus posiciones en IMAGEN: parametrizar asi es estable, mientras que tocar las
# 8 entradas de la matriz directamente no lo es.
CTRL = np.array([[0, 1450], [2015, 5550], [6000, 2585], [6000, 4415]],
                dtype=np.float32)
MAX_DIST_PX = 40.0     # la distancia se trunca: robusto a lineas no detectadas


def line_mask(frame):
    """Mascara de lineas blancas dentro del cesped."""
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # El campo: el componente verde conexo mas grande. Sirve para no tomar como
    # lineas los carteles publicitarios ni la ropa del publico.
    green = ((h > 25) & (h < 95) & (s > 60)).astype(np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(green)
    if n < 2:
        return np.zeros(frame.shape[:2], np.uint8)
    field = (lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))).astype(np.uint8)
    field = cv2.dilate(field, np.ones((9, 9), np.uint8))

    # Filtro de linea: un pixel de linea es mas claro que sus vecinos a AMBOS
    # lados, en horizontal o en vertical. Descarta bordes y sombras grandes.
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    w = 7
    lm = np.zeros_like(g)
    for dx, dy in ((w, 0), (0, w)):
        a = np.roll(g, (dy, dx), (0, 1))
        b = np.roll(g, (-dy, -dx), (0, 1))
        lm = np.maximum(lm, np.minimum(g - a, g - b))
    mask = ((lm > 18) & (field > 0) & (s < 110) & (v > 110)).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _samples(step=25):
    from check_pitch_overlay import pitch_lines
    import numpy as np
    pts = []
    for ln in pitch_lines():
        a = np.array(ln, dtype=np.float32)
        for p, q in zip(a, a[1:]):
            n = max(1, int(np.linalg.norm(q - p) / step))
            pts.extend(p + (q - p) * t for t in np.linspace(0, 1, n, endpoint=False))
    return np.array(pts, dtype=np.float32)


def cost_map(mask):
    """Distancia (truncada) de cada pixel a la linea blanca mas cercana."""
    import cv2
    d = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
    return np.minimum(d, MAX_DIST_PX).astype(np.float32)


def alignment(H, cost, samples=None):
    """Distancia media, en px, entre las lineas proyectadas y las pintadas."""
    import cv2
    if samples is None:
        samples = _samples()
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return float("inf"), 0
    p = cv2.perspectiveTransform(samples.reshape(-1, 1, 2), Hi).reshape(-1, 2)
    ok = (np.isfinite(p).all(1) & (p[:, 0] >= 0) & (p[:, 0] < cost.shape[1] - 1)
          & (p[:, 1] >= 0) & (p[:, 1] < cost.shape[0] - 1))
    if ok.sum() < 50:
        return float("inf"), int(ok.sum())
    q = p[ok]
    return float(cost[np.int32(q[:, 1]), np.int32(q[:, 0])].mean()), int(ok.sum())


def refine(H0, frame, rounds=3, min_samples=400):
    """``(H, antes, despues)``. ``H0`` es la inicializacion (los keypoints)."""
    import cv2
    from scipy.optimize import minimize

    mask = line_mask(frame)
    cost = cost_map(mask)
    S = _samples()
    before, _ = alignment(H0, cost, S)

    def unpack(v):
        H, _ = cv2.findHomography(v.reshape(-1, 2).astype(np.float32), CTRL)
        return H

    def obj(v):
        H = unpack(v)
        if H is None:
            return 1e6
        s, n = alignment(H, cost, S)
        return s + (0.0 if n > min_samples else 5.0)

    v = cv2.perspectiveTransform(
        CTRL.reshape(-1, 1, 2), np.linalg.inv(H0)).reshape(-1, 2).ravel().astype(np.float64)
    for _ in range(rounds):
        v = minimize(obj, v, method="Powell",
                     options={"xtol": 0.05, "ftol": 1e-4,
                              "maxiter": 20000, "maxfev": 20000}).x
    H = unpack(v)
    after, _ = alignment(H, cost, S)
    return (H, before, after) if after < before else (H0, before, before)
