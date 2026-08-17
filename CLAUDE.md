# JC NCF — event tracking automático de fútbol

Contexto para agentes. Lo que sigue es lo que NO se deduce leyendo el código:
qué se probó, qué falló y por qué, y dónde está el problema hoy.

## Objetivo

Armar el departamento de datos del equipo de fútbol de la universidad. Convertir
video de partido en event data automática (pases, pérdidas, recuperaciones,
balón parado, duelos, aéreos, tiros, goles, faltas).

Felipe es estudiante de Data Science y trabaja en el rubro. Las explicaciones
técnicas van completas, sin simplificar de más.

## Estado: qué funciona y qué no

Anda: detección de jugadores y pelota, tracking, generación de eventos por
reglas, herramienta de etiquetado, extracción de features, entrenamiento.

No anda: la **calidad de los eventos**. El clasificador entrenado empata con las
reglas y ambos apenas superan la línea base trivial ("todo es PASE").

## 🔴 Problema abierto — LA HOMOGRAFÍA ES INESTABLE

Descubierto en agosto 2026. Es la causa raíz y todo lo demás está subordinado.

`main.py` re-estima la homografía (imagen→cancha) cada `homography_every`=5
frames. Hasta ahora aceptaba soluciones con **4 keypoints**, que es el mínimo
exacto para resolver una homografía: sin redundancia el error de reproyección es
cero por construcción, así que una solución mala es indetectable. Con puntos
amontonados o casi alineados, la transformación sale exacta cerca de ellos y
disparatada lejos — justo donde está el juego.

Medido sobre spain-france (89.027 frames):

- Un punto que **no se mueve ni 2 px en pantalla** recibe >1 m de desplazamiento
  de cancha en el 4,5% de los casos. **p99 = 25 metros.**
- En el **6,36% de los frames salta todo el plantel junto** más de 1 m; en el
  1,67%, más de 10 m. Los jugadores no se teletransportan en formación.
- **93,6% de esos saltos caen en frames ≡1 (mod 5)** — los de refresco. Al azar
  sería 20%. En el clip de 3 min: 98,4%.

Consecuencia: **todas las coordenadas de cancha están contaminadas**, jugadores
incluidos, y todas las features del clasificador se calculan ahí. Es el
sospechoso principal del techo de AUC 0,6-0,68.

No se arregla post-hoc. Probado: re-alinear bloques consecutivos usando a los
jugadores como anclas acumula deriva (cada corrección arrastra ~14 cm de ruido de
movimiento real, por 17.800 refrescos) → corrección mediana de 10 m, y la pelota
empeora. La alineación incremental no tiene referencia absoluta; sólo los
keypoints la tienen.

Arreglo implementado en `main.py`, **pendiente de validar** (requiere re-track):
`MIN_HOMOGRAPHY_POINTS` 4→6, rechazo de configuraciones degeneradas
(`MIN_KEYPOINT_SPREAD`, `MIN_KEYPOINT_ASPECT` por SVD), rechazo por error de
reproyección (`MAX_REPROJECTION_CM`), y control de continuidad
(`MAX_HOMOGRAPHY_JUMP_CM`): se proyectan los mismos píxeles con la
transformación vieja y la nueva y se conserva la vieja si el mundo se corre
demasiado, salvo que la nueva reproyecte 2x mejor (para no quedar clavado en una
mala).

Diagnóstico: `python3 data_cleanup/check_homography.py --tracking-csv <csv>`.
Mide estabilidad sin ground truth ni video, usando el desplazamiento mediano del
plantel. **Número a batir en el clip de 3 min: 9,12% de frames con salto >1 m.**

## Qué YA se descartó (no repetir)

Ocho intervenciones sin efecto positivo sobre los eventos. Todas operaban río
abajo de las coordenadas rotas, que es probablemente por qué:

| Intervención | Resultado |
|---|---|
| ReID (5701→1912 ids) | eventos idénticos |
| Limpieza de pelota estática | **peor** (borraba la pelota real en pausas) |
| Viterbi global de trayectoria | nulo |
| Ball-crop (pelota 56,9%→83,2%) | AUC 0,678→0,619 |
| Re-etiquetado controlado | precisión de propuestas 12%→10% |
| Veto geométrico del punto de penal (x2) | nunca disparó: los enganches caen a 3,5-4,5 m de la marca nominal |
| Lista negra por celdas (`ball_unpin.py`) | clavados 23,5%→3,8% pero pelota cerca del penal 4,23%→**4,24%** |
| Veto por frame de rachas estáticas | clavados a 0,0% y near-penal **sin cambio** |

Ese último fue el que rompió el caso: si eliminar TODAS las rachas estáticas no
mueve la métrica del síntoma, las detecciones falsas no están quietas — la
cancha se mueve debajo de ellas.

También descartado: **T-DEED / action spotting** para SHOT/GOAL (falla en footage
no-broadcast) y **reescribir la generación de candidatos** (medido: el 100% de
los pases agregados a mano caían a menos de 3 s de una propuesta rechazada; el
recall de MOMENTOS es ~100%, falla la CLASIFICACIÓN).

## Clases de bug recurrentes

1. **El sidecar de fps.** Todo lo que lee el CSV de tracking debe leer
   `<csv>.meta.json` (`effective_fps`, `frame_stride`). Roto y arreglado en 4
   archivos distintos: `match.py`, `event_generator.py`, `reid.py`,
   `label_tool.py`. Los frames del CSV son consecutivos; el frame de VIDEO es
   `(k-1)*stride+1`. Traducir sólo al buscar en el video.
2. **Medir el síntoma, no un proxy.** Pasó cinco veces que la observación visual
   de Felipe le ganó a una medición agregada mía. Cuando reporta que algo se ve
   mal, hay que medir *eso*, no una métrica cercana. Umbrales mal calibrados
   esconden el problema (usé 80 cm/frame como "quieto", que a 15 fps son 12 m/s
   — una pelota rodando normal).
3. **Validar en un clip de 3 min antes de gastar 2,5 h de GPU.** Cuatro
   iteraciones de `_pick_ball` se resolvieron así.
4. **supervision `Detections`** se indexa con SLICE (`[i:i+1]`), nunca escalar:
   valida que `xyxy` sea 2-D.
5. **No inventar negativos.** Un tracking mejor detecta transiciones nuevas sin
   etiqueta; contarlas como negativas le enseña al modelo que los pases reales
   no son pases (AUC 0,678→0,593). `train.py --proposals` las descarta.

## Pipeline

Track en Colab (`event_generation/pipeline_completo.ipynb`, A100) → después todo
CPU local:

```
tracking.csv + tracking.meta.json + tracking_ball_candidates.csv
  → data_cleanup/ball_viterbi.py
  → data_cleanup/reid.py            (re-correr después de CADA re-track)
  → data_cleanup/clean_offpitch.py
  → events_model/propose_events.py
  → events_model/label_tool.py      (etiquetado manual)
  → events_model/train.py
```

Videos en `~/football_data/matches/<partido>/`. Etiquetas versionadas en
`events_model/dataset/`. Rama de trabajo: `events-model`.

Herramientas de medición: `data_cleanup/benchmark.py` (tracking+eventos),
`data_cleanup/check_homography.py` (estabilidad de coordenadas).

## Detalles que muerden

- Modelos distintos a propósito: detector entrenado para jugadores, community
  `football` para la pelota (el entrenado detecta la pelota peor, 54% vs 78%).
- Home/Away son **arbitrarios** (k-means sobre color de camiseta), cambian entre
  corridas. En `label_tool`: naranja = team 0 = Home, azul = team 1 = Away.
- Arqueros: el k-means tiene sólo 2 clusters y les asigna equipo al azar. Se
  marcan como `GOALKEEPER_TEAM_ID = 2` (equipo desconocido) en vez de inventarlo.
  Una heurística por vecinos se probó y **falló**.
- Colab: `pip install -U ultralytics`, reinstalar pillow limpio y reiniciar,
  `numpy<2.1`. La celda de tracking debe BORRAR el CSV viejo y verificar que se
  recreó — si no, una corrida que crashea deja el CSV viejo en Drive y todo el
  pipeline corre sobre datos viejos con resultados byte-idénticos.
- No commitear `tracking.csv` (173 MB) ni `spain-france-eval.zip` (66 MB):
  GitHub rechaza el push.

## Próximo paso

Re-trackear el clip de 3 min (`spain-france-test3min.mp4`, ~5 min de GPU) con el
arreglo de homografía y correr `check_homography.py`. Si el 9,12% baja fuerte,
recién ahí re-trackear el partido completo, re-medir la pelota y re-entrenar —
y ver si el techo de AUC era esto.
