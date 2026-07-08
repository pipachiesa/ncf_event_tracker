"""
Arma el dataset de entrenamiento juntando las features de varios partidos,
con split POR PARTIDO (regla de oro de la fase 3: la generalizacion
cross-match es el criterio; nunca splitear dentro de un clip).

Uso:
    python3 events_model/build_dataset.py \
        events_model/dataset/psg_bayern_features.csv \
        events_model/dataset/psg_inter_features.csv \
        --holdout psg_inter \
        --out-dir events_model/dataset

Escribe train.csv y holdout.csv, e imprime el balance de clases por split
(el desbalance de SHOT/GOAL/FOUL es el que despues se maneja con class
weights / focal loss al entrenar).
"""

import argparse
import csv
import os
from collections import Counter


def load(path):
    with open(os.path.expanduser(path), newline="") as f:
        return list(csv.DictReader(f))


def class_balance(rows):
    return Counter(r["label"] or "(sin label)" for r in rows)


def main():
    ap = argparse.ArgumentParser(description="Split por partido")
    ap.add_argument("features", nargs="+", help="CSVs de features (uno por partido)")
    ap.add_argument("--holdout", required=True,
                    help="match_id reservado para validacion final (no se entrena)")
    ap.add_argument("--out-dir", default="events_model/dataset")
    args = ap.parse_args()

    rows = []
    for path in args.features:
        rows.extend(load(path))

    matches = sorted({r["match_id"] for r in rows})
    if args.holdout not in matches:
        raise SystemExit(f"--holdout {args.holdout!r} no esta en {matches}")

    train = [r for r in rows if r["match_id"] != args.holdout]
    holdout = [r for r in rows if r["match_id"] == args.holdout]
    if not train:
        raise SystemExit("El split dejo el train vacio: hacen falta >= 2 partidos.")

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for name, split in (("train.csv", train), ("holdout.csv", holdout)):
        with open(os.path.join(out_dir, name), "w", newline="\n") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split)

    print(f"partidos: {matches}  (holdout: {args.holdout})")
    for name, split in (("train", train), ("holdout", holdout)):
        print(f"\n{name}: {len(split)} filas")
        for label, count in class_balance(split).most_common():
            print(f"  {label:12s} {count}")

    # TODO(fase 3, tarea 4): entrenar aca (gradient boosting sobre train.csv,
    # validacion leave-one-match-out entre los partidos de train, class weights
    # por el desbalance) y recien al final medir una sola vez contra holdout.csv.


if __name__ == "__main__":
    main()
