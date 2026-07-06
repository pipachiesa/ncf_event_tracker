# ReID — Diseño y contexto (branch `reid`)

## Contexto del proyecto
- **Repo/fork:** `github.com/pipachiesa/ncf_event_tracker` — `main` = pipeline v1 estable; `reid` = este trabajo.
- **Objetivo:** departamento de datos para un equipo de fútbol universitario; automatizar el *event tracking* desde video de **una sola cámara** (estilo broadcast).
- **Pipeline:** `data_cleanup/main.py` (YOLO tracking → CSV raw) → `lib.Match.import_raw_data` → `match.generate_events()` → CSV de eventos. Se corre en Colab con `event_generation/pipeline_completo.ipynb`.
- **Estado v1 (en `main`):** balón ~87%, homografía (coords reales del campo), modelo de jugadores propio entrenado (yolov8l), eventos con distribución realista, clasificador de equipos. (Config de tracking: `--player-model <best.pt entrenado>`, `--ball-model football`, `--imgsz 1280`, `--pitch-model football-field`, etc.)

## Problema que ataca el ReID
`sv.ByteTrack` es solo movimiento (sin apariencia) → **fragmenta**: ~187 IDs distintos para ~25 jugadores (**ratio 7.5×**). Eso rompe las stats por jugador e infla los turnovers (pérdidas/recuperaciones) porque cada ID nuevo/flip parece un cambio de posesión.

### Baseline (clip `psg_bayern_720p`, 5 min, 7172 frames, fps 24)
- **IDs de jugador distintos: 187** (métrica principal a bajar).
- Vida mediana de track: 10.2 s; media 35 s; máx 298 s.
- Tracks >30s: 65 | 5-30s: 53 | <5s (fragmentos): 69 | <1s (basura): 16.
- **Clasificador de equipos ruidoso:** 98/187 tracks cambian de equipo *dentro* del mismo track; reparto 129 vs 58 (debería ser ~50/50).

## Enfoque elegido: **B — merge por apariencia en post-proceso (standalone)**
Módulo aparte (`data_cleanup/reid.py`) que toma **CSV de tracking + video** → **CSV con IDs fusionados**. **NO toca `main.py` ni el loop de tracking.** Los eventos corren sobre el CSV fusionado.
- **Por qué B:** riesgo mínimo (no toca el loop frágil), testeable solo, **cero dependencias nuevas**.
- **Descartados:** A (BoT-SORT nativo de ultralytics) = reescribir el loop + deps nuevas; C (boxmot) = dep nueva. Quedan como plan B si esto no alcanza.

## Embedding de apariencia
- **Histograma de color HSV** del crop del jugador (OpenCV `cv2`, ya es dependencia → **cero deps nuevas**). Promedio de ~8-10 crops muestreados por track (usar `X1,Y1,X2,Y2` del CSV para cropear del video).
- **Limitación honesta:** distingue poco entre **compañeros** (misma camiseta) → se usa como **filtro** (para NO fusionar camisetas distintas), no como señal principal.
- Mejora futura si hace falta: reusar embeddings SigLIP del clasificador de equipos, o OSNet (torchreid).

## Algoritmo de merge (union-find, iterativo)
Fusionar tracks A y B si se cumple **TODO**:
1. **Disjuntos en el tiempo:** A termina antes de que empiece B (sin solapamiento).
2. **Gap corto:** `B.frame_inicio - A.frame_fin ≤ Tmax`.
3. **Plausibilidad espacial:** posición proyectada de A (última pos + velocidad × gap) cerca del inicio de B (`≤ Dmax` cm).
4. **Mismo equipo** (voto mayoritario por track).
5. **Apariencia compatible:** similitud de histograma `≥ Smin`.

Iterar hasta que no haya más merges (un track fusionado se extiende y puede enganchar con otro). A cada track fusionado asignarle **un solo equipo** (voto mayoritario) → bonus: baja turnovers fantasma.
Coords en **cm** (campo 12000×7000). fps 24.

## Prototipo YA probado (solo espacio-temporal + equipo, SIN apariencia)
Sobre el baseline real:
| Settings | IDs | Mal-merges (overlap temporal) |
|---|---|---|
| baseline | 187 | — |
| gap≤2s | 143 | 0 |
| gap≤3s, iterado | 118 | 0 |
| gap≤5s, iterado | 104 | 0 |
| gap≤10s, iterado | **84** | 0 |

La idea funciona y es segura en conteo. **Pero "0 overlaps" es necesario, no suficiente:** con gaps grandes se podría fusionar con el **jugador equivocado del mismo equipo** que pasó cerca — eso lo detecta la **apariencia**, no el overlap. Por eso el filtro de apariencia es clave para la *corrección*, no solo el conteo.

## Tarea a construir
1. **`data_cleanup/reid.py`** con:
   - `load_tracks(csv)` → por track: lista de (frame, x_pitch, y_pitch, bbox), pos inicio/fin, velocidad (de los últimos ~12 frames), equipo mayoritario.
   - `track_descriptor(video, track)` → histograma HSV promedio de ~8-10 crops muestreados (cropea con `X1,Y1,X2,Y2` del CSV).
   - `merge_tracks(...)` → union-find iterativo con los 5 gates.
   - `write_merged_csv(csv, mapping)` → reescribe `Object ID` con el ID fusionado y el equipo mayoritario.
   - CLI: `python data_cleanup/reid.py --tracking-csv X --video Y --output Z [--tmax --dmax --smin]`.
2. **Celda en `pipeline_completo.ipynb`** para correrlo **después del tracking y antes de `generate_events`** (que los eventos usen el CSV fusionado).
3. **Validación:** IDs antes/después, chequear 0 solapamientos, **pintar unos tracks fusionados sobre el video** para verificar visualmente que son el mismo jugador, y re-correr eventos comparando distribución (esperar menos turnovers fantasma).

## Objetivo, riesgo y constraints
- **Objetivo:** 187 → **~40-70** IDs sin sobre-fusionar.
- **Riesgo principal:** sobre-fusionar (juntar dos jugadores distintos, peor que fragmentar). **Preferir quedarse corto.** Gates espacio-temporales duros; apariencia descarta.
- **No se puede testear en el sandbox** (sin GPU/video) → iterar en Colab.
- **Constraints Colab (heredados, ver memoria):** ultralytics reciente (`pip install -U ultralytics`), pillow limpio (`--force-reinstall pillow`), `numpy<2.1` (para numba/umap del clasificador de equipos), no reusar CSV viejo (borrarlo antes de trackear), reiniciar el entorno una vez tras instalar.

## Estado de tareas
1. ✅ Branch `reid` + baseline + clip de validación.
2. ✅ Spike: enfoque B decidido (este doc).
3. ⬜ Extraer descriptores por track (CSV + video).
4. ⬜ Implementar el merge (union-find, 5 gates).
5. ⬜ Tunear umbrales + validar (visual + eventos).
6. ⬜ Integrar en el notebook + comparar + merge a `main`.
