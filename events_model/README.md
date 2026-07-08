# Fase 3: clasificador de eventos entrenado

Ver `EVENTS_CLASSIFIER_DESIGN.md` (raíz del repo) para el diseño completo y
`SPIKE_SOCCERNET.md` para la investigación del spotter de video pre-entrenado.

**División del trabajo:** tracking-features → PASS, TURNOVER, SET PIECE, DUEL.
Video (T-DEED pre-entrenado, ver spike) → SHOT, GOAL, FOUL.

## Flujo de etiquetado (por cada partido)

```bash
# 1. Generar las propuestas de las reglas (las auditadas de events-audit)
python3 events_model/propose_events.py \
    --tracking "~/Downloads/mi_partido.csv" \
    --out dataset/mi_partido_events.csv

# 2. Corregirlas contra el video (aceptar/rechazar/reetiquetar/agregar).
#    Guarda progreso solo (.review.json): se puede cortar y retomar.
#    Al salir con q exporta dataset/mi_partido_events_groundtruth.csv
python3 events_model/label_tool.py \
    --video ~/Desktop/foostats_ai/input_videos/mi_partido.mp4 \
    --events dataset/mi_partido_events.csv \
    --tracking "~/Downloads/mi_partido.csv"

# 3. Features de tracking + labels (una fila por transición de posesión)
python3 events_model/features.py \
    --tracking "~/Downloads/mi_partido.csv" \
    --match-id mi_partido \
    --labels dataset/mi_partido_events_groundtruth.csv \
    --out dataset/mi_partido_features.csv

# 4. Con >= 2 partidos etiquetados: dataset con split POR PARTIDO
python3 events_model/build_dataset.py dataset/*_features.csv \
    --holdout mi_partido_holdout
```

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

Notas: `dataset/` está gitignoreado (datos derivados). Todo esto corre en CPU
local; lo único que pide GPU (Colab) es la inferencia del spotter T-DEED.
