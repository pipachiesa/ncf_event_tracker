# Spike: action spotting pre-entrenado (SoccerNet) para SHOT / GOAL / FOUL

**Fecha:** 2026-07-08 · **Branch:** `events-model` · **Pregunta:** ¿existe un spotter
pre-entrenado, con pesos descargables sin API, que corra sobre nuestro footage broadcast
y cubra shot/goal/foul con poco o cero etiquetado?

**Respuesta corta: sí.** Hay dos modelos viables con checkpoints públicos. El plan de
video se confirma; el etiquetado pesado queda solo para la parte de tracking.

---

## 1. Los dos datasets/tareas de SoccerNet (no confundirlos)

| Tarea | Clases | Nos sirve para |
|---|---|---|
| **Action Spotting (SoccerNet-v2)** | 17: Goal, **Shots on target, Shots off target, Foul**, Corner, Throw-in, Direct/Indirect free-kick, Kick-off, Penalty, Clearance, Ball out of play, Offside, Substitution, tarjetas | **SHOT, GOAL, FOUL** (+ cross-check de SET PIECE) |
| **Ball Action Spotting (2024)** | 12: Pass, Drive, Header, High Pass, Out, Cross, Throw In, **Shot**, Ball Player Block, Player Successful Tackle, Free Kick, **Goal** | segunda opinión de SHOT/GOAL (no tiene FOUL) |

Ambas están anotadas sobre **video broadcast** (720p / 224p) — el mismo tipo de footage
que nuestros clips. El **NDA de SoccerNet aplica solo a los videos del dataset**, no a
los modelos: los checkpoints se bajan libremente.

## 2. Candidatos evaluados

### ✅ Elegido: T-DEED — `github.com/arturxe2/T-DEED`
- **1º puesto en el SoccerNet Ball Action Spotting Challenge 2024** (CVPRW 2024).
- Checkpoints descargables de Google Drive (descarga directa, sin API ni NDA), con
  configs para **SoccerNetBall** (12 clases, incl. Shot/Goal) y **SoccerNet-v2**
  (17 clases, incl. Foul).
- Trae **`inference.py` que acepta un video suelto** y extrae los frames solo — el
  camino más directo para correrlo sobre nuestros clips.
- Corre sobre frames extraídos; **GPU recomendada** → correrlo en el Colab existente.

### ✅ Fallback / opción CPU: E2E-Spot — `github.com/jhong93/spot`
- Del paper "Spotting Temporally Precise, Fine-Grained Events in Video" (ECCV 2022).
- Checkpoints **hosteados directo en GitHub** (`github.com/jhong93/e2e-spot-models`):
  `soccer_rny002gsm_gru_rgb` y `soccer_rny008gsm_gru_rgb`, entrenados en SoccerNet-v2
  (17 clases). Licencia BSD-3.
- Arquitectura liviana (RegNet-Y 200MF/800MF + GRU) → para un clip de ~8 min la
  inferencia en CPU es lenta pero posible; con GPU es trivial.
- Necesita frames pre-extraídos (ffmpeg) — un paso más de plomería que T-DEED.

### ❌ Descartados
- **`lRomul/ball-action-spotting`** (1º del challenge 2023): pesos descargables, pero
  el checkpoint solo detecta **PASS y DRIVE** — no cubre shot/goal/foul.
- **Baselines de `SoccerNet/sn-spotting`** (NetVLAD++, CALF): corren sobre features
  ResNet-152 pre-extraídas, precisión temporal peor que T-DEED/E2E-Spot; solo como
  último recurso.

## 3. Decisión: qué evento va por dónde

| Evento | Fuente | Por qué |
|---|---|---|
| PASS | tracking (modelo propio) | las reglas ya lo proponen; el problema es el parpadeo de posesión → aprendible con features |
| BALL LOST / RECOVERY (turnover) | tracking (modelo propio) | ídem: la señal es la dinámica de posesiones |
| SET PIECE | tracking (modelo propio) | posición del restart + parada larga; el spotter da corner/free-kick/throw-in como **cross-check** |
| DUEL | tracking (modelo propio) | proximidad de dos oponentes + pelota disputada |
| **SHOT** | **video (spotter pretrained)** | el balón no se detecta cerca del arco (4,7 % de frames) → el tracking es ciego justo ahí |
| **GOAL** | **video (spotter pretrained)** | ídem + señales visuales (festejo, replay) que el spotter aprendió |
| **FOUL** | **video (spotter pretrained)** | contacto + reacción del árbitro: invisible para el tracking; las reglas ni lo proponen |

## 4. Riesgos y mitigaciones

- **Precisión temporal:** los spotters dan un timestamp ±unos segundos, no un frame
  exacto. Mitigación: NMS sobre las detecciones + mapear cada detección a la posesión
  del tracking más cercana en el tiempo (frame anchor).
- **Domain gap:** SoccerNet es broadcast europeo; nuestros clips también (psg_bayern,
  psg_inter) → gap chico. Si el footage futuro es de cámara fija amateur, habrá que
  fine-tunear (las etiquetas de shot/goal del labeling tool sirven para eso).
- **GPU:** T-DEED necesita GPU para ser práctico → Colab (ya tenemos flujo). E2E-Spot
  rny002 es el plan B en CPU local.
- **Umbral de confianza:** calibrar contra los clips conocidos: psg_bayern (0 tiros
  reales → no debe disparar) y psg_inter (1 gol real que las reglas perdieron → lo
  tiene que encontrar). Ese par es el smoke test perfecto del spotter.

## 5. Próximo paso de validación (cuando toque Colab)

1. Clonar T-DEED en Colab, bajar el checkpoint SoccerNet-v2 (17 clases).
2. Correr `inference.py` sobre `psg_bayern_720p.mp4` y el clip psg_inter.
3. Criterio de éxito: encuentra el gol de psg_inter, no inventa tiros en psg_bayern
   (recordar: los "tiros" que veían las reglas eran despejes — verificado frame a frame).
