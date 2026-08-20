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

## El arreglo que se probó y NO funcionó (18-ago 12:39)

Acumular keypoints entre refrescos, arrastrándolos con el movimiento de cámara:

- `KeypointBuffer` (15 refrescos = 75 frames = 5 s) guarda `(img_pts, pitch_pts)`.
- `median_image_shift()` estima el paneo con el desplazamiento **mediano** de los
  jugadores en pantalla (cada uno se mueve para su lado, la mediana cancela el
  movimiento propio y deja el de la cámara).
- En cada refresco: `advance(dx,dy)` corre lo guardado, se agregan los keypoints
  nuevos, y se resuelve desde el conjunto acumulado.
- `solve_homography()` exige spread **en cancha** (`MIN_KEYPOINT_PITCH_X_CM=3000`,
  `MIN_KEYPOINT_PITCH_Y_CM=1500`), no sólo en imagen.

### Resultado medido: no sirvió, y dejó el mapa casi congelado

```
Keypoints acumulados al final: 150 puntos cubriendo 20 x 56 m de cancha
Homografia: 29 aceptadas, 486 descartadas por calidad, 25 por salto (94.6% rechazo)
```

150 puntos ÷ 15 refrescos = **10 por refresco**: son los mismos diez keypoints
del diagnóstico del frame 500, quince veces. El span quedó igual que el de un
frame solo.

**La razón es geométrica, no un bug.** `pitch_span()` se mide sobre las
coordenadas de **cancha**, que son etiquetas fijas del vértice detectado (el
vértice 8 es el vértice 8 siempre). Lo que `advance(dx,dy)` corrige son las
coordenadas de **imagen**. Entonces el span en cancha crece **sólo si el modelo
detecta vértices distintos en refrescos distintos**, y a 1,25 px de paneo por
refresco, en los 5 s de la ventana la cámara se movió ~19 px: ve los mismos
diez vértices. El arrastre puede estar perfecto y no cambia nada.

Y como 20 m < `MIN_KEYPOINT_PITCH_X_CM` (30 m), `solve_homography` rechazó casi
todo: **29 mapas aceptados en 540 refrescos**, o sea el mismo modo de falla del
mapa congelado del 17-ago (96,5%). Esa corrida es PEOR que el baseline del
18-ago; no usarla.

El error de fondo fue dimensionar la ventana a ojo. El doc decía "la información
está repartida en el tiempo", que es cierto, pero repartida en **decenas de
segundos** (el frame 1 y el frame 500 están a 33 s), no en 5.

Tests pasados: arrastre correcto, ventana acotada, estimación de cámara,
rechazo del cluster de 20 m. **Se corrieron ad hoc y no quedaron versionados**
(el único test en el repo es `event_generation/test_synthetic.py`). Nótese que
los cuatro pasaban y aún así el enfoque no servía: probaban la mecánica, no la
premisa.

### ⚠️ La corrida del 18-ago 12:39 marcó 19,7% de pelota en el área. NO es un arreglo.

Ese número es exactamente el esperado por superficie (x1,0) y parece el cierre
del problema. No lo es, y el propio bloque CALIBRACION lo dice:

```
x: p01 13.2  p50 56.0  p99  91.9 m   (cancha 0-120)
-> MAL CALIBRADA: nadie aparece nunca en los primeros 13 m; nadie pasa de 92 m
Homografia: 29 aceptadas, 486 descartadas (94.6% rechazo)   <- mapa congelado
```

Con 29 mapas en 540 refrescos el clip corre casi congelado, y los jugadores
ocupan 13-92 m de una cancha de 120. En ese sistema de coordenadas corrido, las
zonas donde el script busca las áreas no son las áreas. La prueba de que el
número es arbitrario: **el mismo tipo de mapa congelado dio 6,1% el 17-ago y
19,7% ahora**. Es ruido de calibración, no señal.

Regla, otra vez: **la métrica del síntoma sólo se lee si CALIBRACION dice
"plausible"**.

### La medición que decide qué hacer

`check_keypoint_coverage.py` (Colab, ~1-2 min sobre el clip): para ventanas de
1 a 120 s calcula la **unión** de vértices detectados y cuánta cancha cubren.
Como las coordenadas de cancha son absolutas, eso es el **techo** de cualquier
acumulación con esa ventana, sin importar qué tan bien ande la compensación del
paneo. Da tres veredictos:

| veredicto | qué hacer |
|---|---|
| "ACUMULAR SIRVE con ventana de N s" | subir `KEYPOINT_BUFFER_REFRESHES` y re-trackear |
| "NINGUNA VENTANA RAZONABLE" | frame de referencia + movimiento de cámara de pocos DOF |
| "NI EL CLIP ENTERO" | el problema es la detección, no la acumulación |

### Resultado de esa medición (18-ago, clip de 3 min)

```
por muestra:  keypoints p50 9    span x p50 20,1 m    max 50,8 m
              muestras que solas pasan el minimo (30 x 15 m): 51 de 540 (9,4%)
vertices distintos en TODO el clip: 17 de 32   (span 69 x 70 m)

ventana     span x p50   % de ventanas >= 30 m
   5 s         20,1 m           22%
  15 s         20,1 m           35%
  30 s         50,8 m           59%
  60 s         50,8 m           84%
 120 s         50,8 m          100%
 clip entero   69,2 m
```

**Veredicto: ACUMULAR SIRVE, con ventana de ~60 s (180 refrescos).** El techo de
este material es 69,2 m: el modelo nunca detecta más de 17 de los 32 vértices.
Alcanza de sobra — el mínimo exigido son 30 m.

### El segundo arreglo (18-ago 13:10): ventana de 60 s + una entrada por vértice

Subir la ventana a 180 refrescos **no alcanza solo**, y esto es lo que casi
cuesta otra vuelta: `ViewTransformer` llama a `cv2.findHomography` **sin método
robusto**, o sea mínimos cuadrados sobre todos los puntos. Con 180 refrescos,
los ~9 vértices del área aparecen 180 veces cada uno y los del círculo central
—los únicos que aportan cancha— unas pocas. El ajuste quedaría determinado en
un 99% por el cluster de 20 m, **y `pitch_span` lo aprobaría igual**, porque
mira el mín/máx y no ve que el resto son duplicados. Sería peor que el bug
original: pasaría todos los controles estando igual de mal.

Por eso el `KeypointBuffer` ahora **indexa por vértice**: una entrada por
vértice, la más reciente, con edad. Cada vértice tiene un voto.

Y el paneo se estima con los **vértices** en vez de con los jugadores: un
vértice de cancha no se mueve nunca, así que su desplazamiento en pantalla es
el de la cámara. Además re-ancla, porque se mide contra la posición guardada:
mientras el vértice siga a la vista, el error de arrastre no se acumula. Los
jugadores quedan de plan B.

Validado en simulación (cámara paneando con homografía conocida, ruido de 2 px,
540 refrescos), midiendo el error de calibración sobre TODA la cancha y no sólo
sobre los puntos del ajuste:

| esquema | rechazadas | error de calibración p50 | p90 |
|---|---|---|---|
| una entrada por vértice | 59 | **45 cm** | 53 cm |
| con duplicados | 30 | 103 cm | 196 cm |

El de duplicados **acepta más y está 2-4x peor**: los duplicados inflan el
conteo de puntos y le hacen pasar el filtro. Los rechazos del esquema nuevo son
todos por "pocos puntos" (4-5 vértices), o sea cuando de verdad falta
información, no por geometría degenerada.

La corrida imprime dos cosas nuevas para verificarlo sobre datos reales: el
**error de arrastre** (cuánto se corrió un vértice viejo cuando reaparece —
sano < 10 px) y el porcentaje de refrescos donde el paneo se estimó con
vértices.

El camino del **frame de referencia**, si esto no alcanza: una homografía son 8 grados de libertad y
un parche de 20 m no los determina. Pero si se resuelve UNA buena con los frames
que sí ven mucha cancha (el frame 1 llega a 49 m porque alcanza el círculo
central), después por refresco alcanza con estimar el movimiento de cámara
relativo — 2 a 4 DOF, que un parche de 20 m sí determina. No acumula deriva
porque cada refresco re-ancla contra keypoints absolutos.

Dato de la geometría, ya verificado: los diez keypoints del área dan 20,2 m de
span; **unidos con los seis del círculo central dan 69,2 m**. O sea que un solo
momento con el centro en cámara alcanza para anclar.

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

## Resultado del segundo arreglo (18-ago 13:40) — el mapa mejoró, el síntoma no

Corrida `..._crop (5).csv`, con ventana de 60 s y un voto por vértice:

| | congelado (4) | baseline (3) | **nueva (5)** |
|---|---|---|---|
| rechazo de homografías | 94,6% | — | **27,6%** |
| saltos >1 m | 0,56%* | 7,04% | **4,23%** |
| `x: p01` | 13,2 | 4,6 | 6,4 |
| `x: p99` | 91,9 | 102,4 | 92,1 |
| `y: p99` (cancha = 70) | — | 76,9 | **68,9** |
| detecciones fuera | 3,4% | 10,5% | **4,3%** |
| pelota en el área | 19,7%* | 58,6% | 50,6% → **45,6% con Viterbi** |

\* números falsos: mapa congelado.

**El mapa dejó de estar congelado** (391 mapas aceptados de 540) y mejoró en
todo lo que se puede verificar sin ground truth. Pero el síntoma sigue.

### La métrica del síntoma, corregida

El 19,7% "esperado por superficie" supone que la pelota se reparte pareja por
la cancha, y en este clip se juega en un solo campo. La referencia correcta son
**los jugadores**, que están sujetos a la misma homografía y al mismo partido:

```
pelota en un area:    45,6%
jugadores en un area:  7,5%     -> la pelota esta ahi 6,1x mas que ellos
x mediana de la pelota:   19 m  |  x mediana de los jugadores: 45 m
```

Esa razón de 6,1x es el síntoma, y **no depende de dónde se haya jugado**. Está
implementada en `check_homography.py`.

### Dónde se engancha: NO es un punto

Con el mapa ya confiable se puede localizar. Sobre 250 celdas de 2x2 m
ocupadas, las 5 más visitadas suman apenas el 15%, y las de arriba están sobre
la **línea de gol y las líneas del área**: `(0,28)`, `(2,34)`, `(4,34)`,
`(16,32)` m. O sea las marcas blancas, repartidas — exactamente lo que Felipe
describió, y la razón por la que todos los vetos "sobre la marca de penal"
fallaron: nunca hubo UNA marca.

⚠️ **Esto reabre el veto de rachas estáticas.** El veto #4 (borrar todas las
rachas estáticas) no movió la métrica y de ahí salió el diagnóstico de la
homografía. Pero eso se midió con el mapa roto: las detecciones falsas no
quedaban quietas porque la cancha se movía debajo. **Ahora sí quedan quietas**:
medido sobre la corrida nueva, 46% de los frames con pelota están en rachas que
se mueven <6 px/frame en pantalla, y en cancha derivan 1-2 m en 6-8 s. El veto
merece re-medirse.

### Sobre "nadie pasa nunca de 92 m"

Ese renglón del bloque CALIBRACION puede ser una **falsa alarma**. La medición
de cobertura mostró que los 17 vértices que el modelo llega a detectar en todo
el clip cubren x ∈ [0, 69] m: la cámara nunca muestra la mitad lejana. Todo lo
que esté más allá de 69 m es extrapolación, y exigir que `x: p99` llegue a 110
supone que el clip recorre la cancha entera. No está verificado que el juego
haya pasado de 92 m. El criterio sirve para un partido completo, no para 3 min
en un solo campo.

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
6. **Una premisa falsa pasa todos los tests unitarios.** El `KeypointBuffer`
   pasó cuatro tests (arrastre, ventana, estimación de cámara, rechazo del
   cluster) y no servía para nada: los tests probaban que la mecánica hacía lo
   que yo quería, no que lo que yo quería sirviera. Antes de implementar,
   medir el TECHO de lo que el enfoque podría dar si funcionara perfecto.
7. **No inventar negativos.** Un tracking mejor detecta transiciones nuevas sin
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
| `check_keypoint_coverage.py` | Techo de cobertura de cancha por ventana de tiempo, y mejores frames de referencia. **Colab.** |
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

1. ~~`check_keypoint_coverage.py`~~ hecho: veredicto "acumular sirve con 60 s".
2. Re-trackear el clip con `KEYPOINT_BUFFER_REFRESHES = 180` y el buffer
   indexado por vértice. Mirar, en este orden: el **error de arrastre**
   (< 10 px), el **rechazo de homografías** (si pasa 80% el mapa está
   congelado y no hay que leer nada más), y recién ahí la calibración.
3. `check_homography.py` sobre el CSV nuevo. Baseline a batir, de la última
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
