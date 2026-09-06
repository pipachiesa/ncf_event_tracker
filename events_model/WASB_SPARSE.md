# Experimento: WASB con etiquetas ralas reales

Implementado el 6-sep-2026. **No hay todavía un resultado de fine-tune ni una
mejora demostrada de detección.** El pipeline de producción no cambia.

La alternativa al etiquetado denso es supervisar solo el frame central anotado:
WASB recibe tres imágenes consecutivas del video original y genera tres mapas,
pero la loss utiliza únicamente el central. Los vecinos aportan contexto sin
recibir targets inventados. Esto permite usar las etiquetas existentes; no
asegura que sean suficientes para generalizar ni resuelve por sí solo oclusiones.

## Qué se corrige

- El exportador XML anterior todavía convertía desconocidos en negativos. Ahora
  rechaza rangos sin anotación completa antes de extraer imágenes. `--max-interp`
  queda obsoleto y solo admite 0. Los notebooks históricos que piden interpolar
  fallarán explícitamente; no deben usarse para este experimento.
- `eval_ball.py` distingue aislamiento comprobado, apoyo temporal y desconocido.
  En el baseline local SF: 242/538 aciertos a 100 px (45%), 54 detecciones reales
  perdidas por el path, **0 aisladas verificadas y 54 indeterminadas**. No hay GT
  vecino suficiente para sostener la afirmación histórica de 100% aisladas.
- El fracaso previo del fine-tune no prueba que sea imposible entrenar con labels
  ralos. Interpolación, negativos falsos, loss, resolución y validación son factores
  distintos; no se estableció experimentalmente una causa única.

## Ejecución

Dependencias del script: torch, opencv-python, hydra-core, omegaconf y las del
repositorio oficial WASB para cargar su modelo. Usar el checkpoint original
`wasb_soccer_best.pth.tar`, que contiene `model_state_dict`.

Primero auditar el partido, con un directorio de salida nuevo:

```sh
python events_model/wasb_sparse.py \
  --video /ruta/brasil_noruega/video.mp4 \
  --labels events_model/dataset/ball_gt/brasil_noruega_ball_labels.csv \
  --meta /ruta/brasil_noruega/tracking.meta.json \
  --out /ruta/experimentos/brasil-audit --audit-only
```

Después entrenar (en GPU, con otro directorio de salida):

```sh
python events_model/wasb_sparse.py \
  --video /ruta/brasil_noruega/video.mp4 \
  --labels events_model/dataset/ball_gt/brasil_noruega_ball_labels.csv \
  --meta /ruta/brasil_noruega/tracking.meta.json \
  --wasb-src /ruta/WASB-SBDT/src \
  --checkpoint /ruta/wasb_soccer_best.pth.tar \
  --out /ruta/experimentos/brasil-sparse-512 \
  --epochs 8 --batch-size 4 --device cuda
```

No extrae PNGs masivos: lee tripletas directamente del video; ahorra disco pero
los seeks pueden ser lentos. Los videos deben ser los originales que corresponden
al CSV y sidecar, sin recortes ni offsets. CSV k -> fuente (k-1)*stride; contexto
fuente -1,0,+1. El primer/último frame sin contexto completo se excluye.

La partición temporal 70/30 usa un embargo mínimo de cinco segundos, sin compartir
imágenes de contexto. `split.json` deja registrados los frames exactos. En Brasil
la auditoría encuentra 991 centros visibles de train y 134 de validación: la
validación es pequeña y no equivale a un test independiente.

La loss MSE balancea zona de pelota y fondo; no se presenta como la loss QFL
original. BatchNorm conserva sus estadísticas preentrenadas. Primero se evalúa
el checkpoint original, y `best.pth` lo conserva si ninguna época mejora acc@100.
`metrics.json` registra acc@20/50/100, precisión condicional a GT visible y falsos
positivos sobre etiquetas invisibles. El umbral predeterminado 0.5 se aplica al
valor del heatmap, **no a una probabilidad calibrada**. Las predicciones de la mejor
época quedan en `validation_predictions.csv` (solo frames evaluados; no tracking
completo ni formato de candidatos para Viterbi).

## Protocolo antes de integrar

1. Correr 512x288 primero para comparar con el modelo original en idénticos datos.
2. Como experimento separado, probar `--width 960 --height 544 --batch-size 2`.
3. No ajustar umbrales mirando el test. La validación elige checkpoint; reservar
   otro bloque/partido como test externo antes de afirmar mejora generalizable.
4. Auditar repeticiones: algunas etiquetas `visible=0` de Brasil significan
   repetición/pausa, no necesariamente ausencia visual. Separarlas requiere una
   anotación semántica adicional; no reinterpretarlas automáticamente.
5. Antes de producción, generar candidatos sobre clips completos, pasar Viterbi
   y comparar contra YOLO+Viterbi en los mismos frames, también cerca de eventos.
   Esta integración y el fine-tune quedan pendientes de medir el experimento.

Verificaciones locales: cuatro tests (indexado de video realista, máscara de
supervisión, embargo y aislamiento), auditoría de etiquetas reales y forward/
backward con HRNet real sobre una imagen de Brasil a 128x96 y pesos aleatorios.
Esa última prueba es de compatibilidad, no mide exactitud ni carga del checkpoint.

Fuentes: https://github.com/nttcom/WASB-SBDT y su
`src/datasets/soccer.py` (el loader XML convierte frames sin target en invisibles).
