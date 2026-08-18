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

## 🔴 ABIERTO — LA HOMOGRAFÍA ESTÁ MAL CALIBRADA (~35-40 m de error)

Distinto del problema de estabilidad de abajo, que sí está resuelto.
`check_homography.py` mide que las coordenadas **no salten**; una transformación
constante y completamente equivocada pasa ese test con nota perfecta.

Evidencia en el clip de 3 min ya "arreglado":

- El arquero, parado dentro de su arco, proyecta a **x = 39,4 m**. La línea de
  gol es x = 0.
- En 3 minutos de juego **ningún jugador aparece nunca en los primeros 35 m**
  (`x: p01 = 34,8`), con el arquero visible en cámara todo el tiempo.
- Jugadores con **y negativo** (−10,5 m), fuera del campo.
- Candidatos a pelota sobre los **carteles publicitarios** (pantalla y≈152, las
  letras blancas de "Booking.yeah") mapean a cancha (40,9, 10,1) — adentro del
  campo, así que el filtro de fuera-de-cancha no los toca. Es lo que Felipe
  reporta: "lejos de la cámara toma puntos blancos de afuera antes que la
  pelota".

⚠️ **Esto invalida toda métrica basada en distancias de cancha**, incluidas las
mías de "pelota cerca de la marca de penal" (daban 0,00% porque el mapa está
corrido, no porque el problema no exista).

**CAUSA IDENTIFICADA (y es un bug introducido al "arreglar" la estabilidad).**
La corrida imprime:

```
Homografia: 19 aceptadas, 84 descartadas por calidad, 437 descartadas por salto (96.5% rechazo)
```

Sólo 19 de 540 aceptadas: el clip entero corre con una transformación calculada
al principio y **congelada**. El control de continuidad comparaba a dónde
proyectan *los mismos píxeles* con la transformación vieja y la nueva — pero
cuando la cámara panea, el mismo píxel corresponde a otro punto del mundo, así
que la transformación **tiene que** cambiar. El filtro castigaba el
comportamiento correcto.

Y un mapa congelado no puede saltar, así que sacaba 0,33% en el test de
estabilidad mientras ponía al arquero a 39 m de su arco. **Optimicé el proxy en
vez del objetivo.** Por eso `check_homography.py` ahora mide las dos cosas por
separado: ESTABILIDAD (que no salte) y CALIBRACIÓN (que apunte a la cancha
correcta, vía el rango de posiciones de los jugadores).

Arreglo: `MAX_HOMOGRAPHY_JUMP_CM` 250 → 1500, o sea guarda anti-catástrofe en
vez de control de continuidad. El trabajo lo hacen los filtros ABSOLUTOS (6+
puntos, spread, error de reproyección), que no congelan nada. **Pendiente de
re-trackear y validar.**

Si tras eso sigue mal calibrada, la otra sospecha es el emparejamiento
keypoint↔vértice: `build_pitch_transformer` hace `pitch_pts = vertices[mask]`,
asumiendo que el keypoint *i* del modelo es el vértice *i* de
`SoccerPitchConfiguration`. Si el modelo usa otro orden u otra geometría, sale
una homografía auto-consistente (error de reproyección bajo) pero apuntando al
lugar equivocado.

Diagnóstico: `data_cleanup/check_pitch_keypoints.py` (correr en Colab, necesita
los modelos). Imprime qué keypoints se usan, el error de reproyección, y a qué
pedazo de cancha proyectan las esquinas de la imagen.

## ✅ ESTABILIDAD DE LA HOMOGRAFÍA — ARREGLADA Y VALIDADA (17-ago-2026)

Validado re-trackeando el clip de 3 min:

| | antes | después |
|---|---|---|
| frames con salto >1 m | 9,12% | **0,33%** |
| frames con salto >5 m | 1,82% | **0** |
| p99 desplazamiento del plantel | 1011 cm | **55 cm** |
| pelota a <4 m de la marca de penal | 5,20% | **0,00%** (por área tocaría 1,41%) |

El síntoma que Felipe venía reportando hace semanas ("agarra el punto de penal")
desapareció. Eventos en el clip: 31 → 47, con PASS 8 → 16.

A/B limpio (mismo `--ball-crop` en las dos corridas, sólo cambia la homografía):
`<4 m de la marca de penal` **5,20% → 0,00%**, presencia tras interpolar
100% → 98,9%.

⚠️ **Ojo con la métrica "clavada"**: subió 10,1% → 14,2%, pero cambió de
significado. Con la homografía rota, una pelota realmente quieta *parecía*
moverse por el jitter; ahora una pelota quieta se ve quieta. No es una regresión.

**`StaticGuard` implementado y QUITADO tras medirlo.** Aislado sobre los
candidatos del clip ya arreglado:

| config | pelota | imposibles | p99 | <4m penal |
|---|---|---|---|---|
| sólo continuidad | 72,0% | 2,26% | 37,8 m/s | **0,00%** |
| + fuera de cancha | 71,5% | 2,28% | 37,8 m/s | **0,00%** |
| + StaticGuard | 71,1% | 4,26% | **704 m/s** | 0,00% |

No aporta nada al síntoma y mete teletransportes: al soltar el ancla, el frame
siguiente re-adquiere por confianza sin límite de distancia (103 sueltas). La
selección actual = continuidad + rechazo fuera de cancha, nada más.

### Cómo era el problema (para no re-introducirlo)

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

Arreglo implementado en `main.py` (**ya validado**, ver tabla arriba):
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

1. `git pull` (el `StaticGuard` salió después de la corrida del clip) y
   re-trackear el partido completo con `BALL_CROP = True` (~2,5 h).
2. Re-correr la cadena (`ball_viterbi` → `reid` → `clean_offpitch`), re-proponer
   eventos y **re-entrenar sobre las etiquetas que ya existen** (están ancladas a
   frames y el video no cambió, así que no hay que re-etiquetar).
4. La pregunta que responde todo esto: **¿el techo de AUC 0,6-0,68 era la
   homografía?** Si sube, las features nunca fueron el problema — estaban
   escritas en un sistema de coordenadas que saltaba metros.
