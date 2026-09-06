# Experimento: ¿el recorte de alta resolución mejora la detección de pelota?

**Pregunta:** hoy el tracking corre a `--imgsz 1280` sobre un video de 1920×1080,
así que YOLO reescala el frame y la pelota pasa de 8,8 px nativos a **5,9 px**.
El recorte procesa una ventana de 640×640 a resolución **nativa**, donde la
pelota conserva sus 8,8 px. ¿Alcanza ese 1,5x para mejorar los eventos?

**Criterio de éxito:** balón presente sube de 56,9% y el AUC del clasificador
sube de 0,678. Si el AUC no se mueve, el techo no es la resolución y hay que ir
por features de video.

---

## 1. Pushear el código (en tu Mac)

```bash
cd ~/Desktop/FootballTrackingDataGeneration
git add -A
git commit -m "ball: recorte de alta resolucion + arqueros equipo 2"
git push origin events-model
```

Sin esto Colab levanta el `main.py` viejo y el experimento no prueba nada.

## 2. Cortar el clip de prueba

```bash
brew install ffmpeg     # solo si no lo tenés

ffmpeg -ss 00:26:00 \
  -i /Users/felipechiesa/football_data/matches/spain-france/video.mp4 \
  -t 00:03:00 -c copy \
  /Users/felipechiesa/Downloads/spain-france-test3min.mp4
```

`-c copy` copia los frames sin recomprimir: cero pérdida de calidad, que es
justo lo que estamos midiendo. El minuto 26 es un tramo que ya etiquetaste.

## 3. Subir a Google Drive

Subí a `MyDrive/football_analytics/videos/`:

    spain-france-test3min.mp4

Y subí a Colab el notebook actualizado:

    /Users/felipechiesa/Desktop/FootballTrackingDataGeneration/event_generation/pipeline_completo.ipynb

## 4. Correr las DOS versiones en Colab

En la celda del video:

```python
VIDEO_FILE = 'spain-france-test3min.mp4'
```

**Corrida A (referencia).** En la celda de tracking:

```python
BALL_CROP = False
```

Ejecutá hasta la celda 5b y anotá `frames con pelota` e `IMPOSIBLES`.

**Corrida B (con recorte).** Cambiá a:

```python
BALL_CROP = True
```

Volvé a ejecutar la celda de tracking y la 5b.

Los CSV salen con sufijos distintos (`_base` y `_crop`), así que no se pisan.

## 5. Bajar los resultados

De `MyDrive/football_analytics/tracking_output/` bajá los cuatro archivos:

    spain-france-test3min_base.csv
    spain-france-test3min_base.meta.json
    spain-france-test3min_crop.csv
    spain-france-test3min_crop.meta.json

Guardalos en `/Users/felipechiesa/football_data/test/` (creá la carpeta).

## 6. Comparar (en tu Mac)

```bash
cd ~/Desktop/FootballTrackingDataGeneration

python3 data_cleanup/benchmark.py \
  --tracking /Users/felipechiesa/football_data/test/spain-france-test3min_base.csv \
  --label "SIN recorte"

python3 data_cleanup/benchmark.py \
  --tracking /Users/felipechiesa/football_data/test/spain-france-test3min_crop.csv \
  --label "CON recorte"
```

**Qué mirar:** `presente` tiene que subir y `IMPOSIBLES` mantenerse bajo. Si el
balón sube y los imposibles no, el recorte sirve.

---

## Si el clip da bien: partido completo

En Colab, `VIDEO_FILE = 'spain-france.mp4'` y `BALL_CROP = True`. Son ~2,5 h
(menos que las 4 h de la corrida actual, porque el recorte reemplaza trabajo
más caro).

Después, en tu Mac, la cadena completa:

```bash
cd ~/Desktop/FootballTrackingDataGeneration

# 1. trayectoria global
python3 data_cleanup/ball_viterbi.py \
  --tracking-csv /Users/felipechiesa/football_data/matches/spain-france/tracking.csv \
  --output       /Users/felipechiesa/football_data/matches/spain-france/tracking_vit.csv

# 2. identidad de jugadores
python3 data_cleanup/reid.py \
  --tracking-csv /Users/felipechiesa/football_data/matches/spain-france/tracking_vit.csv \
  --video        /Users/felipechiesa/football_data/matches/spain-france/video.mp4 \
  --output       /Users/felipechiesa/football_data/matches/spain-france/tracking_reid.csv

# 3. sacar suplentes y pelotas del costado
python3 data_cleanup/clean_offpitch.py \
  --tracking-csv /Users/felipechiesa/football_data/matches/spain-france/tracking_reid.csv \
  --output       /Users/felipechiesa/football_data/matches/spain-france/tracking_onpitch.csv

# 4. reentrenar sobre las MISMAS etiquetas que ya tenés
python3 events_model/train.py \
  --tracking /Users/felipechiesa/football_data/matches/spain-france/tracking_onpitch.csv \
  --labels events_model/dataset/spain-france_labeled.csv \
  --labels events_model/dataset/spain-france_10m_18m_proposed_groundtruth.csv \
  --labels events_model/dataset/spain-france_26m_34m_proposed_groundtruth.csv \
  --labels events_model/dataset/spain-france_64m_72m_proposed_groundtruth.csv \
  --proposals events_model/dataset/spain-france_proposed.csv
```

No hay que re-etiquetar nada: las etiquetas están ancladas a frames y el video
es el mismo.

**El número que decide: AUC.**

| AUC | Lectura | Qué hacer |
|-----|---------|-----------|
| 0,75+ | la resolución era el techo | etiquetar más, ahora sí rinde |
| 0,70-0,75 | mejora parcial | seguir, pero mirar también video |
| ~0,678 | la resolución NO era el techo | ir por features de video |
