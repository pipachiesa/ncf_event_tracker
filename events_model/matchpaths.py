"""
Convencion de rutas para el flujo por partido de la fase 3.

Datos crudos (pesados, fuera del repo):
    ~/football_data/matches/<match>/video.mp4
    ~/football_data/matches/<match>/tracking.csv

Etiquetas + features (livianos, se commitean):
    events_model/dataset/<match>_proposed.csv   (propuestas de las reglas)
    events_model/dataset/<match>.review.json    (progreso del label_tool)
    events_model/dataset/<match>_labeled.csv    (ground truth exportado)
    events_model/dataset/features.csv           (tabla de features combinada)

Los scripts (propose_events.py, label_tool.py, features.py, build_dataset.py)
resuelven estas rutas solos con --match <nombre>; --tracking/--video/--events/
--out siguen aceptando rutas explicitas para casos fuera de la convencion.
"""

import os

FOOTBALL_DATA_DIR = os.path.expanduser("~/football_data/matches")
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


def match_dir(match):
    return os.path.join(FOOTBALL_DATA_DIR, match)


def tracking_path(match):
    return os.path.join(match_dir(match), "tracking.csv")


def video_path(match):
    return os.path.join(match_dir(match), "video.mp4")


def proposed_path(match):
    return os.path.join(DATASET_DIR, f"{match}_proposed.csv")


def review_path(match):
    return os.path.join(DATASET_DIR, f"{match}.review.json")


def labeled_path(match):
    return os.path.join(DATASET_DIR, f"{match}_labeled.csv")


def features_path(match):
    return os.path.join(DATASET_DIR, f"{match}_features.csv")
