"""
Arma el dataset de entrenamiento juntando las features de varios partidos,
con split POR PARTIDO (regla de oro de la fase 3: la generalizacion
cross-match es el criterio; nunca splitear dentro de un clip).

Uso (convencion por partido, extrae features de cada --match y splitea):
    python3 events_model/build_dataset.py \
        --match psg_bayern --match psg_inter --match psg_lyon \
        --val psg_lyon --test psg_inter

Con un solo --match (smoke test / partido en etiquetado) corre igual: todo
va a train.csv y val/test quedan vacios.

Uso (rutas explicitas a features ya calculadas, modo legado):
    python3 events_model/build_dataset.py \
        events_model/dataset/psg_bayern_features.csv \
        events_model/dataset/psg_inter_features.csv \
        --test psg_inter

Escribe features.csv (tabla combinada, solo en modo --match), train.csv,
val.csv y test.csv, e imprime el balance de clases por split (el desbalance
de SHOT/GOAL/FOUL es el que despues se maneja con class weights / focal loss
al entrenar).
"""

import argparse
import csv
import os
import sys
from collections import Counter

import matchpaths
from features import extract, FIELDNAMES  # noqa: E402


def load(path):
    with open(os.path.expanduser(path), newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def class_balance(rows):
    return Counter(r["label"] or "(sin label)" for r in rows)


def main():
    ap = argparse.ArgumentParser(description="Split por partido (train/val/test)")
    ap.add_argument("features", nargs="*",
                    help="CSVs de features ya calculadas (modo de rutas explicitas)")
    ap.add_argument("--match", action="append", default=[],
                    help="nombre de partido (convencion ~/football_data/matches/<match>/ "
                         "+ events_model/dataset/<match>_labeled.csv); repetible")
    ap.add_argument("--test", default=None,
                    help="match_id reservado para el split de test (no se entrena)")
    ap.add_argument("--val", default=None,
                    help="match_id reservado para el split de validacion (opcional)")
    ap.add_argument("--holdout", default=None,
                    help="alias legado de --test")
    ap.add_argument("--out-dir", default=matchpaths.DATASET_DIR)
    args = ap.parse_args()

    test_match = args.test or args.holdout
    out_dir = os.path.expanduser(args.out_dir)

    rows = []
    if args.match:
        for m in args.match:
            tracking = matchpaths.tracking_path(m)
            labels = matchpaths.labeled_path(m)
            if not os.path.exists(labels):
                sys.exit(f"Falta {labels}: etiqueta {m} con label_tool.py --match {m} primero.")
            if not os.path.exists(tracking):
                sys.exit(f"Falta el tracking crudo: {tracking}")
            match_rows = extract(tracking, m, labels)
            print(f"{m}: {len(match_rows)} transiciones ({tracking})")
            rows.extend(match_rows)
        os.makedirs(out_dir, exist_ok=True)
        write_csv(os.path.join(out_dir, "features.csv"), rows, FIELDNAMES)

    for path in args.features:
        rows.extend(load(path))

    if not rows:
        sys.exit("No hay features: pasa --match <partido> o CSVs de features ya calculadas.")

    matches = sorted({r["match_id"] for r in rows})
    for name, m in (("--test", test_match), ("--val", args.val)):
        if m and m not in matches:
            sys.exit(f"{name} {m!r} no esta en {matches}")

    held_out = {m for m in (test_match, args.val) if m}
    train = [r for r in rows if r["match_id"] not in held_out]
    val = [r for r in rows if r["match_id"] == args.val] if args.val else []
    test = [r for r in rows if r["match_id"] == test_match] if test_match else []
    if not train:
        sys.exit("El split dejo el train vacio: hacen falta partidos fuera de --val/--test.")

    os.makedirs(out_dir, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for name, split in (("train.csv", train), ("val.csv", val), ("test.csv", test)):
        write_csv(os.path.join(out_dir, name), split, fieldnames)

    print(f"\npartidos: {matches}  (val: {args.val or '-'}  test: {test_match or '-'})")
    for name, split in (("train", train), ("val", val), ("test", test)):
        print(f"\n{name}: {len(split)} filas")
        for label, count in class_balance(split).most_common():
            print(f"  {label:12s} {count}")

    if len(matches) < 2:
        print("\nAviso: con un solo partido no hay split real de val/test todavia "
              "(hacen falta >= 2-3 partidos etiquetados).")

    # TODO(fase 3, tarea 4): entrenar aca (gradient boosting sobre train.csv,
    # validacion contra val.csv por epoch/iteracion, class weights por el
    # desbalance) y recien al final medir una sola vez contra test.csv.


if __name__ == "__main__":
    main()
