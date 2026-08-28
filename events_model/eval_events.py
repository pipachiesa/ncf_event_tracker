#!/usr/bin/env python3
"""Evaluador reproducible del clasificador de EVENTOS (por ahora PASS).

Separado de train.py (que entrena/guarda modelo): esto solo MIDE, con validacion
leave-one-block-out, y reporta las metricas que importan para una herramienta de
revision, no solo ROC-AUC:
  - ROC-AUC (ranking, ciego al desbalance)
  - PR-AUC (average precision) — mas honesto con clases desbalanceadas
  - precision @ recall objetivo (p.ej. 70%): cuanta basura ves si conservas 70%
    de los pases -> traduce directo a tiempo de revision
  - baseline trivial (todo PASS)
  - por bloque, y global

⚠️ n chico (un partido ~296 filas): diferencias <0,03 son ruido. El criterio de
oro es leave-one-MATCH-out (varios partidos); con uno solo esto es orientativo.

Uso:
    python3 events_model/eval_events.py \\
        --tracking ~/football_data/matches/spain-france-recal/tracking_recal_vit.csv \\
        --labels events_model/dataset/spain-france_10m_18m_proposed_groundtruth.csv \\
        --labels events_model/dataset/spain-france_26m_34m_V2_groundtruth.csv \\
        --labels events_model/dataset/spain-france_64m_72m_proposed_groundtruth.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as feat
from train import build_rows, load_label_blocks, to_xy

NOT_FEATURES = {"match_id", "label", "start_frame", "end_frame", "block", "y"}


def roc_auc(scored):
    pos = [s for s, y in scored if y == 1]
    neg = [s for s, y in scored if y == 0]
    if not pos or not neg:
        return float("nan")
    w = sum(1 for a in pos for b in neg if a > b)
    t = sum(1 for a in pos for b in neg if a == b)
    return (w + 0.5 * t) / (len(pos) * len(neg))


def pr_curve(scored):
    """Devuelve (precisions, recalls) barriendo el umbral, y average precision."""
    scored = sorted(scored, key=lambda x: -x[0])
    P = sum(1 for _s, y in scored if y == 1)
    if P == 0:
        return [], [], float("nan")
    tp = fp = 0
    precs, recs = [], []
    ap = 0.0
    prev_rec = 0.0
    for s, y in scored:
        if y == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        rec = tp / P
        precs.append(prec)
        recs.append(rec)
        ap += prec * (rec - prev_rec)      # average precision (area PR)
        prev_rec = rec
    return precs, recs, ap


def prec_at_recall(scored, target=0.70):
    precs, recs, _ = pr_curve(scored)
    best = 0.0
    for p, r in zip(precs, recs):
        if r >= target:
            best = max(best, p)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking", required=True)
    ap.add_argument("--labels", action="append", required=True)
    ap.add_argument("--target", default="PASS")
    ap.add_argument("--recall", type=float, default=0.70)
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    blocks = load_label_blocks(args.labels)
    rows = build_rows(os.path.expanduser(args.tracking), blocks, args.target, reviewed=None)
    cols = [c for c in feat.FIELDNAMES if c not in NOT_FEATURES]
    pos = sum(r["y"] for r in rows)
    print(f"\n{len(rows)} transiciones  ({pos} {args.target} / {len(rows)-pos} no-{args.target})"
          f"  en {len(blocks)} bloques")
    if pos < 20:
        print("⚠ muy pocos positivos: inestable.")

    scored = []
    for name, _lo, _hi, _l in blocks:
        tr = [r for r in rows if r["block"] != name]
        te = [r for r in rows if r["block"] == name]
        if not te or not tr or sum(r["y"] for r in tr) < 5:
            continue
        Xtr, ytr = to_xy(tr, cols)
        Xte, yte = to_xy(te, cols)
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        scored.extend(zip(p, yte))

    _pr, _rc, apv = pr_curve(scored)
    base = pos / len(rows)
    print(f"\nMETRICAS (leave-one-block-out)")
    print(f"  ROC-AUC                 {roc_auc(scored):.3f}   [0,50 azar; <0,60 sin senal]")
    print(f"  PR-AUC (avg precision)  {apv:.3f}   [baseline (prevalencia) {base:.3f}]")
    print(f"  precision @ recall {int(args.recall*100)}%   {prec_at_recall(scored, args.recall):.3f}"
          f"   (conservando {int(args.recall*100)}% de los {args.target})")
    print(f"  baseline trivial (todo {args.target}): precision {base:.3f}")
    print("\nLectura para producto: si a recall 70% la precision es 0,X, de cada 100")
    print("eventos autoaceptados ~ (1-0,X)*100 son basura -> eso es el tiempo de revision.")


if __name__ == "__main__":
    main()
