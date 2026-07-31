# Dataset del clasificador de eventos (fase 3)

Esta carpeta SE COMMITEA (labels + features son livianos). El raw pesado
(video + tracking) vive fuera del repo, en `~/football_data/matches/<match>/`.

## Flujo por partido

```bash
# 0. Trackear el partido (Colab) y guardar el resultado en la convencion:
mkdir -p ~/football_data/matches/<match>
cp video.mp4      ~/football_data/matches/<match>/video.mp4
cp tracking.csv   ~/football_data/matches/<match>/tracking.csv

# 1. Propuestas de las reglas (auditadas, ver events-audit) -> intermedio, no se commitea
python3 events_model/propose_events.py --match <match>

# 2. Corregir contra el video (aceptar/rechazar/reetiquetar/agregar).
#    Progreso autosave en <match>.review.json (se puede cortar y retomar).
#    Al salir con q exporta <match>_labeled.csv -> ESTO SI SE COMMITEA
python3 events_model/label_tool.py --match <match>

# 3. Con >= 1 partido etiquetado: features + split train/val/test por partido
python3 events_model/build_dataset.py \
    --match <match1> --match <match2> --match <match3> \
    --val <match2> --test <match3>

# 4. Commitear las labels (y opcionalmente features.csv)
git add events_model/dataset/<match>_labeled.csv events_model/dataset/features.csv
git commit -m "labels: <match>"
```

## Que se commitea y que no

| Archivo                          | Se commitea | Por que |
|-----------------------------------|:-----------:|---------|
| `<match>_labeled.csv`             | si          | ground truth, no se puede regenerar sin re-etiquetar |
| `features.csv`                    | si          | requiere el tracking crudo (fuera del repo) para regenerarse |
| `<match>_proposed.csv`            | no          | intermedio, se regenera con `propose_events.py --match <match>` |
| `<match>.review.json`             | no          | progreso de sesion del label_tool |
| `<match>_features.csv`            | no          | salida suelta de `features.py --match` (inspección/inferencia), no la tabla combinada |
| `train.csv` / `val.csv` / `test.csv` | no       | se regeneran desde `features.csv` con `build_dataset.py` |

El video y el tracking crudo (`.mp4`, `tracking.csv`) nunca deben terminar
adentro del repo — viven en `~/football_data/matches/<match>/`. El
`.gitignore` tiene una red de seguridad para esto, pero no confies solo en
eso: no copies el raw a `events_model/dataset/`.

## Split train/val/test

Es por partido, no por jugada (regla de oro de la fase 3: la generalizacion
cross-match es el criterio). Con un solo partido etiquetado, `build_dataset.py`
igual corre: todo va a `train.csv` y `val`/`test` quedan vacios hasta tener
mas partidos.
