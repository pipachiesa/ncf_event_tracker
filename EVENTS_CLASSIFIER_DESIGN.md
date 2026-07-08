# Clasificador de Eventos — Diseño (fase 3)

## Contexto
- **Repo/fork:** `github.com/pipachiesa/ncf_event_tracker`. Pipeline: `data_cleanup/main.py` (tracking) → `lib.Match` → `match.generate_events()` → CSV de eventos.
- **Estado:** el tracking v1 es bueno (balón ~78-87%, homografía, modelo entrenado). Pero la clasificación de eventos **por reglas** (`lib/event_generator.py`) **NO generaliza**: validada en un 2º clip (psg-inter) dio **254 pases** (~3× lo esperado para ~8 min) y **perdió un gol real** (0 tiros/goles).
- **Diagnóstico (medido):** los pases inflados NO son por fragmentación de IDs (0 self-passes tras merge) sino por **parpadeo de posesión** (~1.1 s por posesión = imposible). Los tiros/goles se pierden porque el balón **no se detecta cerca del arco** (4.7% de frames). Conclusión: techo de las reglas sobre tracking de una sola cámara.

## Decisión: enfoque HÍBRIDO
Cada falla tiene causa distinta → se ataca donde se puede:
- **Tracking-features** (posición/posesión): **PASS, TURNOVER (BALL LOST / RECOVERY), SET PIECE, DUEL.** Un modelo aprende a ignorar los micro-cambios que las reglas cuentan de más.
- **Video action-spotting** (mira los frames): **SHOT, GOAL, FOUL.** Cosas que el tracking no ve (balón cerca del arco, contacto, reacción del árbitro).

## Atajo clave para la parte de video
Existen modelos **pre-entrenados de SoccerNet action spotting** (SoccerNet-v2: ~17 clases incluyendo goal, shot, foul, corner, etc.). **Investigar en el spike si uno funciona sobre tu footage** → los tiros/goles/faltas podrían salir con **poco o cero etiquetado**. Esto de-riska mucho el proyecto.

## Taxonomía a etiquetar
PASS, DUEL, SET PIECE (corner / throw-in / goal-kick / kick-off / free-kick), FOUL, BALL LOST / RECOVERY (turnover), SHOT, GOAL. (Expandible; arrancar por los factibles.)

## Etiquetado (el cuello de botella real)
- **Estrategia: corregir las propuestas de las reglas**, no etiquetar de cero (5-10× más rápido). La herramienta muestra el evento propuesto + el video, y el humano acepta / rechaza / reetiqueta / agrega (para FOUL, que las reglas no proponen).
- **Varios partidos distintos** (no un solo clip — esa fue la causa del sobreajuste).
- Tiros/goles/faltas son raros → etiquetar aparte y priorizados (o cubrirlos con el spotter pre-entrenado).
- **Reparto de trabajo:** Claude Code construye la herramienta; **Felipe pone las etiquetas**.

## Pipeline ML
- **Features de tracking** (por posesión/ventana): trayectoria del balón, posición/velocidad de jugadores, distancias al arco, densidad local, secuencia de cambios de posesión, duración, etc.
- **Video:** clips cortos alrededor de cada evento (para el spotter).
- **Split train/val/test SEPARADO POR PARTIDO** (no dentro de un clip) — la generalización cross-match es el criterio.
- **Modelos:** tracking → gradient boosting (rápido, interpretable) o NN temporal (1D-CNN / Transformer sobre la secuencia); video → SoccerNet spotter (pretrained o fine-tune).
- **Desbalance:** tiros/goles/faltas raros → class weights / focal loss / oversampling.

## Validación
Partido **held-out** que el modelo no vio. Precision/recall **por tipo**. Foco en lo que falla hoy: pases (que no se inflen 3×) y tiros/goles (que se detecten). Éxito = **generaliza entre partidos**.

## Integración
Reemplazar/aumentar `generate_events` con el modelo. Mismo formato de CSV de salida.

## Realidad y riesgos (honesto)
- Es un proyecto de **semanas**, no horas. El etiquetado es el costo real.
- **FOUL es el más difícil** (contacto + decisión del árbitro) → depende fuerte del video/SoccerNet.
- **Empezar CHICO:** pocos partidos, pocos tipos (arrancar por PASS + TURNOVER con tracking, SHOT/GOAL con el spotter pretrained), **validar el ciclo completo end-to-end**, y recién ahí escalar tipos y partidos.

## Tareas (la roca)
1. Spike: enfoque híbrido confirmado; ¿spotter SoccerNet pretrained sirve? taxonomía + métrica.
2. Etiquetado por corrección de reglas, multi-partido (herramienta + Felipe etiqueta).
3. Feature extraction / prep de datos (split por partido).
4. Modelo + entrenamiento (cross-match, manejar desbalance).
5. Validación en partido held-out (precision/recall por tipo).
6. Integrar en el pipeline (mismo CSV).

## Primer paso concreto para Claude Code
(1) Spike escrito: investigar modelos SoccerNet action-spotting pretrained (¿corren sobre broadcast? ¿qué clases? ¿pesos sin API?) y decidir qué eventos van por video vs tracking. (2) Construir la **herramienta de etiquetado** (propuestas de reglas + video, aceptar/rechazar/reetiquetar/agregar, exporta ground truth). (3) Esqueleto de feature extraction de tracking. Con eso Felipe empieza a etiquetar.
