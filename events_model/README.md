# Fase 3: clasificador de eventos entrenado

Ver `EVENTS_CLASSIFIER_DESIGN.md` (raíz del repo) para el diseño completo y
`SPIKE_SOCCERNET.md` para la investigación del spotter de video pre-entrenado.

**División del trabajo:** tracking-features → PASS, TURNOVER, SET PIECE, DUEL.
Video (T-DEED pre-entrenado, ver spike) → SHOT, GOAL, FOUL.

## Flujo de etiquetado (por cada partido)

Convención de rutas (ver `events_model/matchpaths.py`): el raw pesado
(video + tracking) vive fuera del repo en `~/football_data/matches/<match>/`;
las labels y features livianas se commitean en `events_model/dataset/` (ver
`events_model/dataset/README.md` para el detalle de qué se commitea).

```bash
# 0. Trackear (Colab) y guardar el resultado en la convención
mkdir -p ~/football_data/matches/mi_partido
cp video.mp4    ~/football_data/matches/mi_partido/video.mp4
cp tracking.csv ~/football_data/matches/mi_partido/tracking.csv

# 1. Generar las propuestas de las reglas (las auditadas de events-audit)
python3 events_model/propose_events.py --match mi_partido

# 2. Corregirlas contra el video (aceptar/rechazar/reetiquetar/agregar).
#    Guarda progreso solo (.review.json): se puede cortar y retomar.
#    Al salir con q exporta events_model/dataset/mi_partido_labeled.csv
python3 events_model/label_tool.py --match mi_partido

# 3. Con >= 1 partido etiquetado: features + dataset con split POR PARTIDO
#    (build_dataset.py llama a features.py internamente por cada --match)
python3 events_model/build_dataset.py \
    --match mi_partido --match otro_partido \
    --val otro_partido --test un_tercer_partido
```

Rutas explícitas (sin la convención `--match`) siguen funcionando en los
tres scripts — ver el `--help` de cada uno.

Teclas del label tool: `y` acepta · `x` rechaza · `r`+clase reetiqueta ·
`a`+clase+equipo agrega (p. ej. FOUL, que las reglas no proponen) · `u` undo ·
`n`/`p` evento siguiente/anterior · espacio pausa · `,`/`.` frame a frame ·
`h`/`l` ±1 s · `b` cajas on/off · `q` guarda y exporta. Clases: 1 PASS ·
2 BALL LOST · 3 RECOVERY · 4 SHOT · 5 GOAL · 6 FOUL · 7 SET PIECE · 8 DUEL.

## Estado y próximos pasos

- [x] Spike spotter de video (`SPIKE_SOCCERNET.md`) — T-DEED elegido, correr en Colab
- [x] Herramienta de etiquetado (`label_tool.py`)
- [x] Feature extraction v1 (`features.py`) + split por partido (`build_dataset.py`)
- [ ] Felipe etiqueta 2+ partidos (empezar por psg_bayern y psg-inter)
- [ ] Validar T-DEED en Colab: debe encontrar el gol de psg-inter y no inventar tiros en psg_bayern
- [ ] Entrenar gradient boosting sobre las transiciones (empezar PASS + TURNOVER)
- [ ] Validación en partido held-out, precision/recall por tipo
- [ ] Integrar al pipeline (mismo formato CSV que `generate_events`)

Notas: `dataset/` SÍ se versiona (labels + features, ver
`events_model/dataset/README.md`); los intermedios de trabajo (propuestas,
progreso de review, splits train/val/test) están gitignoreados y se
regeneran. Todo esto corre en CPU local; lo único que pide GPU (Colab) es la
inferencia del spotter T-DEED.
