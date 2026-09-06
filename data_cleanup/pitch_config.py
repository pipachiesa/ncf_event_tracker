"""Geometria de cancha del proyecto. UNICA fuente de verdad.

⚠️ POR QUE ESTE ARCHIVO EXISTE, Y NO SE USA ``SoccerPitchConfiguration``

La configuracion de la libreria ``sports`` que uso todo el pipeline hasta el
20-ago NO describe una cancha reglamentaria:

    |                     | sports    | reglamento |
    | cancha              | 120 x 70  | 105 x 68   |
    | area de penal fondo | 20,15 m   | 16,5 m     |
    | area de penal ancho | 41,0 m    | 40,32 m    |
    | area chica          | 5,5x18,32 | igual ✓    |
    | punto de penal      | 11 m      | igual ✓    |
    | circulo central     | 9,15 m    | igual ✓    |

LA PRUEBA, que no necesita ninguna calibracion: la D es el arco de 9,15 m
centrado en el punto de penal, que esta a 11 m de la linea de gol. Con el area
a 20,15 m, como 11 + 9,15 = 20,15, la D queda EXACTAMENTE TANGENTE al borde del
area y no sobresale nada. En el video la D sobresale, como en cualquier cancha
de futbol. Medido sobre el frame 761 proyectando el circulo completo: con la
geometria de ``sports`` quedan 2 puntos de 200 fuera del area; con la
reglamentaria, 60.

Verificado ademas ajustando la homografia con cada geometria y midiendo la
distancia a las lineas PINTADAS, por elemento:

    geometria           total  linea de gol  area chica b  circulo
    sports 120x70        20,4      3,0          13,6         8,2
    reglamento 105x68    16,7      0,6           0,0         3,0
    area chica en 120x70 22,7      6,9           0,9        11,4

El area chica pasa de 13,6 a 0,0: ese residuo se habia atribuido a oclusion y
era geometria equivocada. La tercera fila muestra que no alcanza con corregir
el area: hace falta tambien el tamaño de la cancha.

QUE INVALIDA de todo lo medido antes del 20-ago
  * las distancias en cancha salian ~14% infladas a lo largo y ~3% a lo ancho;
  * "pelota dentro del area" usaba x < 20,15 m, un 22% mas profunda que la
    real, asi que la fraccion venia inflada;
  * cualquier umbral en cm de cancha (velocidades, radios) hay que re-leerlo.

EL ORDEN DE LOS VERTICES SE MANTIENE. El modelo de keypoints predice el vertice
i-esimo de la lista de ``sports``, que es un landmark con significado (la
esquina del area, la tangencia con la D, ...). Reconstruir la MISMA lista con
las medidas correctas conserva el significado y arregla las coordenadas.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class PitchConfig:
    """Cancha reglamentaria, en centimetros."""

    length: int = 10500          # 105 m (UEFA lo exige en competencia)
    width: int = 6800            # 68 m
    penalty_box_length: int = 1650      # 16,5 m desde la linea de gol
    penalty_box_width: int = 4032       # 7,32 del arco + 16,5 a cada lado
    goal_box_length: int = 550          # 5,5 m
    goal_box_width: int = 1832          # 18,32 m
    penalty_spot_distance: int = 1100   # 11 m
    centre_circle_radius: int = 915     # 9,15 m
    goal_width: int = 732               # 7,32 m entre postes

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """Los 32 vertices, en el MISMO orden que ``SoccerPitchConfiguration``.

        El orden importa: el modelo de keypoints predice el vertice i-esimo de
        esa lista, asi que cambiarlo romperia el emparejamiento.
        """
        L, W = self.length, self.width
        PB, PBW = self.penalty_box_length, self.penalty_box_width
        GB, GBW = self.goal_box_length, self.goal_box_width
        SP, R = self.penalty_spot_distance, self.centre_circle_radius
        return [
            (0, 0),
            (0, (W - PBW) / 2),
            (0, (W - GBW) / 2),
            (0, (W + GBW) / 2),
            (0, (W + PBW) / 2),
            (0, W),
            (GB, (W - GBW) / 2),
            (GB, (W + GBW) / 2),
            (SP, W / 2),
            (PB, (W - PBW) / 2),
            (PB, (W - GBW) / 2),
            (PB, (W + GBW) / 2),
            (PB, (W + PBW) / 2),
            (L / 2, 0),
            (L / 2, W / 2 - R),
            (L / 2, W / 2 + R),
            (L / 2, W),
            (L - PB, (W - PBW) / 2),
            (L - PB, (W - GBW) / 2),
            (L - PB, (W + GBW) / 2),
            (L - PB, (W + PBW) / 2),
            (L - SP, W / 2),
            (L - GB, (W - GBW) / 2),
            (L - GB, (W + GBW) / 2),
            (L, (W - PBW) / 2),
            (L, (W - GBW) / 2),
            (L, (W + GBW) / 2),
            (L, (W + PBW) / 2),
            (L, 0),
            (L, W),
            (L / 2 - R, W / 2),
            (L / 2 + R, W / 2),
        ]

    def polylines(self, n=40):
        """Las lineas pintadas, como listas de puntos (para dibujar o alinear)."""
        L, W = self.length, self.width
        PB, PBW = self.penalty_box_length, self.penalty_box_width
        GB, GBW = self.goal_box_length, self.goal_box_width
        SP, R = self.penalty_spot_distance, self.centre_circle_radius
        import math

        def seg(a, b):
            return [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
                    for i in range(n + 1)]

        y0, y1 = (W - PBW) / 2, (W + PBW) / 2
        g0, g1 = (W - GBW) / 2, (W + GBW) / 2
        out = {
            "linea de gol izq": seg((0, 0), (0, W)),
            "linea de gol der": seg((L, 0), (L, W)),
            "lateral lejano": seg((0, 0), (L, 0)),
            "lateral cercano": seg((0, W), (L, W)),
            "linea media": seg((L / 2, 0), (L / 2, W)),
            "area izq a": seg((0, y0), (PB, y0)),
            "area izq b": seg((PB, y0), (PB, y1)),
            "area izq c": seg((PB, y1), (0, y1)),
            "area der a": seg((L, y0), (L - PB, y0)),
            "area der b": seg((L - PB, y0), (L - PB, y1)),
            "area der c": seg((L - PB, y1), (L, y1)),
            "chica izq a": seg((0, g0), (GB, g0)),
            "chica izq b": seg((GB, g0), (GB, g1)),
            "chica izq c": seg((GB, g1), (0, g1)),
            "chica der a": seg((L, g0), (L - GB, g0)),
            "chica der b": seg((L - GB, g0), (L - GB, g1)),
            "chica der c": seg((L - GB, g1), (L, g1)),
            "circulo central": [
                (L / 2 + R * math.cos(t), W / 2 + R * math.sin(t))
                for t in [i * 2 * math.pi / 72 for i in range(73)]
            ],
        }
        # Las dos "D": el trozo del arco de 9,15 m que queda FUERA del area.
        # Con la geometria de ``sports`` esto seria un punto; que en la cancha
        # real sea un arco visible es la prueba de que aquella estaba mal.
        for side, sx in (("izq", SP), ("der", L - SP)):
            pts = []
            for i in range(121):
                t = -math.pi / 2 + math.pi * i / 120
                x = sx + R * math.cos(t) * (1 if side == "izq" else -1)
                y = W / 2 + R * math.sin(t)
                if (x >= PB) if side == "izq" else (x <= L - PB):
                    pts.append((x, y))
            if len(pts) > 1:
                out[f"D {side}"] = pts
        return out


PITCH = PitchConfig()

# Atajos, para no repetir PITCH.x por todos lados.
PITCH_L_CM = float(PITCH.length)
PITCH_W_CM = float(PITCH.width)
PBOX_L_CM = float(PITCH.penalty_box_length)
PBOX_W_CM = float(PITCH.penalty_box_width)
PBOX_Y0 = (PITCH_W_CM - PBOX_W_CM) / 2
PBOX_Y1 = (PITCH_W_CM + PBOX_W_CM) / 2
PBOX_AREA_FRAC = 2 * PBOX_L_CM * PBOX_W_CM / (PITCH_L_CM * PITCH_W_CM)


def in_penalty_box(x_cm, y_cm):
    """¿Cae (x, y) dentro de alguna de las dos areas de penal?"""
    return (PBOX_Y0 <= y_cm <= PBOX_Y1
            and (x_cm <= PBOX_L_CM or x_cm >= PITCH_L_CM - PBOX_L_CM))
