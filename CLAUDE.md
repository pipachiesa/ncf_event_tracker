# JC NCF — event tracking automático de fútbol

Contexto para agentes. Esto es lo que NO se deduce leyendo el código: qué se
probó, qué falló, por qué, y dónde está el problema hoy. Varias secciones
documentan **errores míos** (del agente) que costaron días; están acá para que
no se repitan.

## Objetivo

Armar el departamento de datos del equipo de fútbol de la universidad. Convertir
video de partido en event data automática: pases, pérdidas, recuperaciones,
balón parado, duelos, aéreos, tiros, goles, faltas.

Felipe es estudiante de Data Science y trabaja en el rubro. Las explicaciones
técnicas van completas, sin simplificar de más. Habla español (Argentina).

## Estado en una línea

El pipeline entero corre de punta a punta, pero **las coordenadas de cancha
están mal calibradas**, y como todas las features del clasificador se calculan
ahí, nada río abajo puede evaluarse en serio hasta cerrar eso.

---

# 🔴 EL PROBLEMA ABIERTO: CALIBRACIÓN DE LA HOMOGRAFÍA

## El síntoma, en las palabras de Felipe

> "cuando transcurre lejos de la cámara, toma el punto penal y hasta puntos
> blancos de afuera de la cancha antes que la pelota"

Lo reportó durante semanas. Yo medí seis veces y seis veces le dije que estaba
arreglado. **Las seis veces tenía razón él.** Ver "Lecciones" al final.

## Qué está roto exactamente

`main.py` estima la homografía (imagen→cancha) desde los keypoints que detecta
`martinjolif/yolo-football-pitch-detection` (32 vértices de
`SoccerPitchConfiguration`, cancha de **120 × 70 m**, punto de penal a
**1100 cm** de la línea de gol).

**Diagnóstico decisivo** (`data_cleanup/check_pitch_keypoints.py`, corrido en
Colab sobre 4 frames del clip):

```
frame 500    umbral  kps  span x (m)  span y (m)  reproy (cm)
                0.5   10        20.1        55.5           93   <- muy poca cancha
                0.1   11        20.1        55.5          102   <- muy poca cancha
  indices usados: 0,1,2,6,7,8,9,10,11,12   -> TODOS del área de penal izquierda
  esquinas de la imagen proyectadas: (-17768,-14706), (2584, 62716) cm
```

En 3 de 4 frames, los únicos keypoints confiables son los del área de penal
izquierda: un parche de **20 m** sobre una cancha de 120. El error de
reproyección es bajo (36-93 cm) porque el ajuste es bueno *ahí*; después
extrapola al resto de la imagen y se va a **cientos de metros**.

**Bajar el umbral de confianza NO ayuda**: a 0.1 el span sigue en 20,1 m. No es
que los keypoints estén poco confiables, es que la cámara no muestra más cancha.
El frame 1, que alcanza a ver el círculo central, llega a 49 m de span con 36 cm
de error. O sea: la información existe pero está **repartida en el tiempo**.

Consecuencias medidas en el clip de 3 min. ⚠️ **Estos cuatro números salen de
la corrida con el mapa CONGELADO** (ver la sección de estabilidad), que ya se
destrabó: son la foto del peor caso, no del estado actual del código.

- El arquero, parado dentro de su arco, proyecta a **x = 39,4 m** (la línea de
  gol es x = 0).
- `x: p01 = 34,8 m` — en 3 minutos ningún jugador aparece nunca en los primeros
  35 m, con el arquero en cámara todo el tiempo.
- Jugadores con **y negativo** (−10,5 m).
- Los candidatos a pelota sobre los **carteles publicitarios** (pantalla y≈152,
  las letras blancas de "Booking.yeah") mapean a cancha (40,9 , 10,1) —
  *adentro* del campo, así que el filtro de fuera-de-cancha no los toca.

Después de destrabar y suavizar, `x: p01` bajó a **4,6 m**, o sea la
calibración mejoró mucho respecto de esa foto. Lo que sostiene que el problema
sigue abierto **no** son esos números sino el diagnóstico de keypoints de arriba
(span de 20 m), que se mide sobre los keypoints crudos y no depende de qué
homografía se haya aceptado. Los otros tres (arquero, y negativo, carteles)
**no se volvieron a medir después del arreglo**: hacerlo es parte del paso 2.

⚠️ **Esto invalida toda métrica basada en distancias de cancha.** Varias
mediciones mías dieron "0,00%" no porque el problema no existiera sino porque el
mapa estaba corrido.

## El arreglo en curso (implementado, PENDIENTE DE VALIDAR)

Acumular keypoints entre refrescos, arrastrándolos con el movimiento de cámara:

- `KeypointBuffer` (15 refrescos = 75 frames = 5 s) guarda `(img_pts, pitch_pts)`.
- `median_image_shift()` estima el paneo con el desplazamiento **mediano** de los
  jugadores en pantalla (cada uno se mueve para su lado, la mediana cancela el
  movimiento propio y deja el de la cámara).
- En cada refresco: `advance(dx,dy)` corre lo guardado, se agregan los keypoints
  nuevos, y se resuelve desde el conjunto acumulado.
- `solve_homography()` exige spread **en cancha** (`MIN_KEYPOINT_PITCH_X_CM=3000`,
  `MIN_KEYPOINT_PITCH_Y_CM=1500`), no sólo en imagen.

La corrida imprime `Keypoints acumulados al final: N puntos cubriendo XX x YY m`.
**Si el span llega a 60-100 m funcionó; si sigue en ~20 m, la compensación del
paneo no está agarrando.**

Tests pasados: arrastre correcto, ventana acotada, estimación de cámara,
rechazo del cluster de 20 m. **Se corrieron ad hoc y no quedaron versionados**
(el único test en el repo es `event_generation/test_synthetic.py`), así que no
se pueden re-correr: si tocás `KeypointBuffer` o `median_image_shift`, hay que
reescribirlos.

## La métrica que SÍ captura el síntoma

**Pelota dentro de un área de penal: 58,6%.** Las dos áreas son el 19,6% de la
superficie (`penalty_box` 4100 x 2015 cm x 2, sobre 12000 x 7000) → **3x
sobre-representada**. Todas mis métricas anteriores (radio de 4 m alrededor de
la marca) eran ciegas, porque los falsos positivos caen a 6-14 m de la marca
nominal. Usar ésta.

Está implementada en `check_homography.py` (bloque `PELOTA DENTRO DE UN AREA DE
PENAL`), y reproduce el 58,6% exacto sobre la corrida del 18-ago 09:44.

⚠️ **Sólo se lee si la calibración es plausible.** Sobre la corrida con el mapa
congelado da **6,1%**, que parece buenísimo y no significa nada: con todo
corrido 35 m a la derecha, el área izquierda cae en coordenadas donde no hay
nada. Mirar siempre el bloque CALIBRACION primero.

---

# 🟡 ESTABILIDAD DE LA HOMOGRAFÍA — muy mejorada, no cerrada

No la llamo resuelta: quedó en **7,04% de frames con saltos >1 m**, y todo se
midió **con el mapa todavía descalibrado**, así que hay que re-medirla cuando
cierre la calibración. Lo que sí está resuelto es la causa catastrófica (los
saltos >5 m pasaron de 1,41% a 0,26%).

Problema original, medido sobre spain-france (89.027 frames):

- Un punto que **no se mueve ni 2 px en pantalla** recibe >1 m de desplazamiento
  de cancha el 4,5% de las veces; **p99 = 25 metros**.
- En el **6,36% de los frames salta todo el plantel junto** más de 1 m (1,67%,
  más de 10 m). Los jugadores no se teletransportan en formación.
- **93,6% de esos saltos caen en frames ≡1 (mod 5)**, los de refresco; al azar
  sería 20%. En el clip de 3 min, 98,4%.

Causa: se aceptaban soluciones con `MIN_HOMOGRAPHY_POINTS = 4`, el mínimo exacto
para resolver una homografía. Sin redundancia el error de reproyección es cero
por construcción y una solución mala es indetectable.

### ⚠️ El error caro: congelé el mapa optimizando mi propia métrica

Primer "arreglo": control de continuidad que comparaba a dónde proyectan **los
mismos píxeles** con la transformación vieja y la nueva, rechazando la nueva si
diferían más de 250 cm. Resultado:

```
Homografia: 19 aceptadas, 84 descartadas por calidad, 437 descartadas por salto (96.5% rechazo)
```

19 de 540: el clip entero corriendo con un mapa calculado al principio y
**congelado**. El razonamiento estaba mal de raíz: **cuando la cámara panea, el
mismo píxel corresponde a otro punto del mundo, así que la transformación TIENE
que cambiar.** El filtro castigaba el comportamiento correcto.

Y un mapa congelado no puede saltar, así que sacaba **0,33%** en mi test de
estabilidad mientras ponía al arquero a 39 m de su arco. Construí la métrica y
después optimicé contra ella. Por eso `check_homography.py` ahora mide dos cosas
separadas: **ESTABILIDAD** (que no salte) y **CALIBRACIÓN** (que apunte a la
cancha correcta, vía el rango de posiciones de los jugadores).

### Lo que funcionó

| | congelado | destrabado | + suavizado |
|---|---|---|---|
| saltos >1 m | 0,33%* | 8,71% | **7,04%** |
| saltos >5 m | 0%* | 1,41% | **0,26%** |
| p99 desplazamiento del plantel | 55 cm* | 630 cm | **265 cm** |
| `x: p01` (debería ser ~0) | 34,8 m | 4,2 m | 4,6 m |
| pelota fuera de cancha | — | 5,1% | **0,8%** |

\* números falsos, salían del congelamiento.

1. `MAX_HOMOGRAPHY_JUMP_CM` 250 → **1500** (guarda anti-catástrofe, no control de
   continuidad). Destrabó la calibración.
2. **Suavizado temporal** (`HOMOGRAPHY_SMOOTH_ALPHA = 0.35`): promedio móvil
   sobre las posiciones de cancha a las que proyectan cuatro puntos fijos de la
   imagen, reconstruyendo la transformación desde ahí (promediar matrices es
   numéricamente feo). Validado en simulación: temblor p99 25→7 cm con 17 cm de
   retraso, a un paneo de 1,25 px por refresco (el real).

**Dato importante: los filtros de calidad NO sirven contra el temblor.** 6 puntos
mínimos + spread + error de reproyección llevaron 9,12% → 8,71%. Todo el efecto
que parecía venir de ellos era el congelamiento. Lo que corta el temblor es el
suavizado.

### No se arregla post-hoc

Probado: re-alinear bloques consecutivos usando a los jugadores como anclas
acumula deriva (cada corrección arrastra ~14 cm de ruido de movimiento real, por
17.800 refrescos) → corrección mediana de 10 m, y la pelota empeora. La
alineación incremental no tiene referencia absoluta; sólo los keypoints la tienen.

---

# La pelota: qué se probó y qué quedó

Selección actual en `_pick_ball`: **gate de continuidad** (radio físico en cm,
`MAX_BALL_SPEED_MS=40`, crece con el hueco) **+ rechazo duro fuera de cancha**
(`BALL_OFFPITCH_MARGIN=0.04`). Nada más.

## Cuatro vetos del "punto de penal", los cuatro fallidos

| Intento | Por qué falló |
|---|---|
| Sobre la marca **Y** sin moverse respecto al frame anterior | el enganche EMPIEZA con un salto desde la pelota real; ese primer frame pasaba |
| N frames seguidos sobre la marca (radio 1,2 m, geometría) | nunca disparó: los enganches caen a **3,5-4,5 m** de la marca nominal |
| Lista negra de celdas sobre-visitadas (`ball_unpin.py`) | clavados 23,5%→3,8% pero pelota cerca del penal 4,23%→**4,24%** |
| Veto por frame de **todas** las rachas estáticas | clavados a **0,0%** y la métrica del síntoma **sin cambio** |

El cuarto rompió el caso: si eliminar TODAS las rachas estáticas no mueve la
métrica, las detecciones falsas **no están quietas** — la cancha se mueve debajo
de ellas. De ahí salió el diagnóstico de la homografía.

`ball_unpin.py` queda en el repo pero está **superado**: atacaba un síntoma.

## `StaticGuard`: implementado y quitado tras medirlo

Soltaba el ancla de continuidad tras N frames quieto. Aislado sobre los
candidatos del clip ya arreglado:

| config | pelota | imposibles | p99 | <4m penal |
|---|---|---|---|---|
| sólo continuidad | 72,0% | 2,26% | 37,8 m/s | 0,00% |
| + fuera de cancha | 71,5% | 2,28% | 37,8 m/s | 0,00% |
| + StaticGuard | 71,1% | 4,26% | **704 m/s** | 0,00% |

No aporta nada al síntoma y **mete teletransportes**: al soltar el ancla, el
frame siguiente re-adquiere por confianza sin límite de distancia (103 sueltas).
No re-agregarlo sin volver a medir el p99.

## El enganche que sí es real (y sigue abierto)

Frame 49 del clip, datos crudos:

```
elegido:  conf 0.311  pantalla (1531, 380)   <- dentro del área
ignorado: conf 0.535  pantalla (1270, 619)   <- casi seguro la pelota real
```

Eligió el de **menor** confianza porque ya venía enganchado y el candidato bueno
estaba a **32 m** — fuera del radio físico (2,67 m/frame). El enganche se
auto-alimenta y el gate causal no puede salir solo.

**`ball_viterbi.py` sí resuelve ese frame** (elige la trayectoria coherente
(1384,604)→(1356,605)→(1306,613)→...). Resolver la trayectoria global por
programación dinámica evita la trampa causal. Está en el pipeline y hay que
correrlo: **no saltearlo** (yo lo salteé al armar comandos y costó una vuelta
entera).

---

# Qué YA se descartó (no repetir)

| Intervención | Resultado |
|---|---|
| ReID (5701→1912 ids) | eventos idénticos |
| Limpieza de pelota estática | **peor** (borraba la pelota real en pausas) |
| Ball-crop (pelota 56,9%→83,2%) | AUC 0,678→0,619 |
| Re-etiquetado controlado | precisión de propuestas 12%→10% |
| Re-alineación post-hoc de coordenadas | deriva de 10 m, pelota peor |
| T-DEED / action spotting para SHOT/GOAL | falla en footage no-broadcast |
| Reescribir la generación de candidatos | el recall de MOMENTOS ya es ~100% |

Sobre el último: el **100%** de los pases agregados a mano caían a menos de 3 s
de una propuesta rechazada. Falla la CLASIFICACIÓN, no la generación.

⚠️ Casi todas estas se midieron **con las coordenadas rotas**. Varias pueden
merecer una segunda medición una vez cerrada la calibración — el Viterbi ya
demostró que sirve ahora y antes lo había marcado como nulo.

---

# Lecciones (errores míos, no del código)

1. **Medir el síntoma, no un proxy.** Seis veces la observación visual de Felipe
   le ganó a una medición agregada mía. Casos concretos:
   - Umbral de "quieto" en 80 cm/frame, que a 15 fps son **12 m/s** — o sea una
     pelota rodando normal contada como inmóvil.
   - Punto de penal calculado como `11/105*12000 = 1257 cm` cuando la config es
     de 120 m y el valor correcto es **1100 cm**.
   - Promediar el **módulo** del vector de cámara en vez del vector, que nunca se
     cancela y fabricó un hallazgo falso (16,2% → 7,0% al corregirlo).
   - Métrica de "cerca del penal" con radio de 4 m cuando el falso positivo cae a
     6-14 m: daba 0,00% de forma sistemática.
2. **No optimizar contra una métrica propia sin un chequeo absoluto.** Congelar
   el mapa maximizaba mi test de estabilidad y destruía la calibración.
3. **Validar en el clip de 3 min antes de gastar 2,5 h de GPU.** Cuatro
   iteraciones de `_pick_ball` y tres de la homografía se resolvieron así.
4. **El sidecar de fps.** Todo lo que lee el CSV debe leer `<csv>.meta.json`
   (`effective_fps`, `frame_stride`). Roto y arreglado en 4 archivos distintos:
   `match.py`, `event_generator.py`, `reid.py`, `label_tool.py`. Los frames del
   CSV son consecutivos; el frame de VIDEO es `(k-1)*stride+1`.
5. **supervision `Detections`** se indexa con SLICE (`[i:i+1]`), nunca escalar.
6. **No inventar negativos.** Un tracking mejor detecta transiciones nuevas sin
   etiqueta; contarlas como negativas enseña que los pases reales no son pases
   (AUC 0,678→0,593). `train.py --proposals` las descarta.

---

# Pipeline

Track en Colab (`event_generation/pipeline_completo.ipynb`, A100) → después todo
CPU local:

```
tracking.csv + tracking.meta.json + tracking_ball_candidates.csv
  → data_cleanup/ball_viterbi.py     (NO saltear: resuelve enganches causales)
  → data_cleanup/reid.py             (re-correr después de CADA re-track)
  → data_cleanup/clean_offpitch.py
  → events_model/propose_events.py
  → events_model/label_tool.py       (etiquetado manual)
  → events_model/train.py
```

Videos en `~/football_data/matches/<partido>/`. Etiquetas versionadas en
`events_model/dataset/`. Rama de trabajo: `events-model`.
Fork: `https://github.com/pipachiesa/ncf_event_tracker`.

## Herramientas de diagnóstico

| Script | Qué mide |
|---|---|
| `check_homography.py` | ESTABILIDAD (que no salte) **y** CALIBRACIÓN (que apunte bien). Sin ground truth ni video. |
| `check_pitch_keypoints.py` | Qué keypoints se usan, cuánta cancha cubren, error de reproyección. **Correr en Colab** (necesita los modelos). |
| `benchmark.py` | tracking + eventos con las mismas métricas siempre |

---

# Detalles que muerden

- Modelos distintos a propósito: detector entrenado para jugadores, community
  `football` para la pelota (el entrenado la detecta peor, 54% vs 78%).
- Home/Away son **arbitrarios** (k-means sobre color de camiseta) y cambian entre
  corridas. En `label_tool`: naranja = team 0 = Home, azul = team 1 = Away.
- Arqueros: el k-means tiene 2 clusters y les asigna equipo al azar. Se marcan
  `GOALKEEPER_TEAM_ID = 2` (equipo desconocido). Una heurística por vecinos se
  probó y **falló**.
- El notebook de Colab **se desincroniza del repo**. Pasó: `BALL_CROP` existía
  sólo en la celda que Felipe editaba, así que `--ball-crop` nunca se pasaba y el
  A/B no medía lo que creíamos. Verificar siempre que el `cmd` impreso tenga los
  flags esperados.
- La celda de tracking debe BORRAR el CSV viejo y verificar que se recreó. Si no,
  una corrida que crashea deja el CSV viejo en Drive y todo el pipeline corre
  sobre datos viejos con resultados byte-idénticos.
- Colab: `pip install -U ultralytics`, reinstalar pillow limpio y reiniciar,
  `numpy<2.1`.
- No commitear `tracking.csv` (173 MB) ni `spain-france-eval.zip` (66 MB).
- En Downloads conviven varias versiones del mismo CSV (`(1)`, `(2)`, `(3)`).
  Usar `ls -t ... | head -1`, y confirmar con `check_homography.py` cuál es cuál.
  Identificadas al 18-ago (todas del clip de 3 min, todas con `--ball-crop`):

  | archivo | saltos >1 m | `x: p01` | pelota en área | qué es |
  |---|---|---|---|---|
  | `spain-france-test3min_crop.csv` (12-ago) | 9,12% | −6,3 | 53,4% | antes de los filtros de calidad |
  | `..._crop (1).csv` y `spain-france-test3min.csv` (17-ago) | 0,33% | 34,8 | 6,1% | **mapa CONGELADO** (números falsos) |
  | `..._crop (2).csv` (18-ago 09:16) | 8,71% | 4,2 | 52,6% | destrabado |
  | `..._crop (3).csv` (18-ago 09:44) | 7,04% | 4,6 | **58,6%** | destrabado + suavizado ← **la última buena** |

- ⚠️ `~/football_data/matches/clip-test/tracking.csv` (y sus derivados `_vit`,
  `_clean`, `_interp`) son la corrida del **mapa congelado**, o sea están
  viejos. Todo lo que se corra sobre ellos mide la configuración equivocada.
  Reemplazarlos con el re-track nuevo antes de seguir la cadena.

---

# Próximos pasos

1. Re-trackear el clip de 3 min con la acumulación de keypoints. Mirar
   `Keypoints acumulados: N puntos cubriendo XX x YY m`. **Objetivo: 60-100 m.**
2. `check_homography.py` sobre el CSV nuevo. Baseline a batir, de la última
   corrida buena (`..._crop (3).csv`, 18-ago 09:44):

   | | hoy | objetivo |
   |---|---|---|
   | `x: p01` (arquero en cámara) | 4,6 m | ~0 |
   | `x: p99` | 102,4 m | ~120 |
   | detecciones fuera de los límites | 11% | <5% |
   | saltos >1 m | 7,04% | menos |
   | **pelota dentro del área** | **58,6%** | **~19,7%** |

   El bloque CALIBRACION tiene que pasar de "MAL CALIBRADA" a "plausible".
3. La métrica del síntoma es la última fila: es la que traduce lo que Felipe ve.
4. Recién ahí: partido completo (~2,5 h), cadena de limpieza completa, y
   **re-entrenar sobre las etiquetas que ya existen** (están ancladas a frames y
   el video no cambió, no hay que re-etiquetar).
5. La pregunta que responde todo esto: **¿el techo de AUC 0,6-0,68 era la
   homografía?** Si sube, las features nunca fueron el problema — estaban
   escritas en un sistema de coordenadas equivocado.
