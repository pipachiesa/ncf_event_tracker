"""Entrena el clasificador de eventos sobre features de tracking (MVP: PASS).

QUE HACE
Toma un tracking y uno o varios CSV de etiquetas (los ``*_groundtruth.csv`` /
``*_labeled.csv`` que exporta label_tool), extrae una fila de features por
transicion de posesion, y entrena un gradient boosting para decidir si esa
transicion es un PASE o no.

POR QUE ESTO Y NO MAS REGLAS
Medido sobre etiquetas reales: las reglas proponen un evento en practicamente
todos los momentos correctos (el 100% de los pases agregados a mano caian a
menos de 3 s de una propuesta), pero CLASIFICAN mal. Aflojar los umbrales no
ayuda: sube el recall y hunde la precision. Lo que falta no son mas
candidatos, es decidir mejor cuales son pases -- justo lo que aprende un
modelo con las features del tracking.

DOS DECISIONES METODOLOGICAS QUE IMPORTAN

1. Solo se usan las transiciones que caen DENTRO de un bloque etiquetado. El
   resto del partido no esta revisado: contarlo como "no es pase" seria
   inventar negativos y el modelo aprenderia basura.

2. La validacion es LEAVE-ONE-BLOCK-OUT, nunca aleatoria. Dos transiciones
   consecutivas del mismo tramo comparten jugadores, camara y contexto: un
   split al azar las reparte entre train y test y el modelo aprueba por
   memorizar. Entrenar en unos bloques y validar en OTRO es la unica medida
   honesta con un solo partido; con varios partidos, el split correcto pasa a
   ser por PARTIDO.

Uso:
    python3 events_model/train.py \\
        --tracking ~/football_data/matches/spain-france/tracking.csv \\
        --labels events_model/dataset/spain-france_labeled.csv \\
        --labels events_model/dataset/spain-france_10m_18m_proposed_groundtruth.csv \\
        --labels events_model/dataset/spain-france_26m_34m_proposed_groundtruth.csv
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "data_cleanup"))

import features as feat  # noqa: E402

# Columnas que no son features (identificadores o la etiqueta misma).
NOT_FEATURES = {"match_id", "label", "start_frame", "end_frame"}

# Margen (frames) alrededor del rango etiquetado: una transicion que empieza
# justo en el borde del bloque igual esta revisada.
BLOCK_PAD = 30


def load_label_blocks(paths):
    """[(nombre, frame_min, frame_max, [(frame, tipo), ...])] por archivo."""
    blocks = []
    for path in paths:
        rows = list(csv.DictReader(open(os.path.expanduser(path))))
        if not rows:
            continue
        labels = []
        for r in rows:
            # Los REJECTED no son eventos: son los negativos que buscamos.
            if r.get("Verdict") == "REJECTED":
                continue
            labels.append((int(float(r["Start Frame"])), r["Type"]))
        frames = [int(float(r["Start Frame"])) for r in rows]
        blocks.append((os.path.basename(path).replace("_proposed_groundtruth.csv", "")
                       .replace("_labeled.csv", ""),
                       min(frames) - BLOCK_PAD, max(frames) + BLOCK_PAD, labels))
    return blocks


def build_rows(tracking, blocks, target):
    """Filas de features restringidas a los bloques etiquetados, con su bloque."""
    rows = feat.extract(tracking, "match", labels_csv=None)
    all_labels = [lab for _n, _a, _b, labs in blocks for lab in labs]
    rows = feat.attach_labels(rows, all_labels)

    out = []
    for row in rows:
        for name, lo, hi, _labs in blocks:
            if lo <= row["start_frame"] <= hi:
                row["block"] = name
                row["y"] = 1 if row["label"] == target else 0
                out.append(row)
                break
    return out


def to_xy(rows, cols):
    X = [[float(r[c]) if r[c] not in ("", None) else 0.0 for c in cols] for r in rows]
    y = [r["y"] for r in rows]
    return X, y


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return 100 * p, 100 * r, 100 * f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking", required=True)
    ap.add_argument("--labels", action="append", required=True,
                    help="CSV de ground truth; repetible (uno por bloque)")
    ap.add_argument("--target", default="PASS")
    ap.add_argument("--proposals", default=None,
                    help="CSV de propuestas de las reglas. Permite medir a las "
                         "REGLAS sobre las MISMAS transiciones que el modelo: "
                         "sin esto la comparacion mezcla unidades (eventos "
                         "matcheados por tiempo vs clasificacion por transicion) "
                         "y no dice nada.")
    ap.add_argument("--exclude-block", action="append", default=[],
                    help="Bloque a excluir (p.ej. el primero, etiquetado "
                         "mientras aprendias la herramienta).")
    ap.add_argument("--out", default=None, help="donde guardar el modelo (.joblib)")
    args = ap.parse_args()

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError:
        sys.exit("Falta scikit-learn: pip3 install scikit-learn")

    blocks = load_label_blocks(args.labels)
    print(f"bloques etiquetados: {len(blocks)}")
    for name, lo, hi, labs in blocks:
        print(f"  {name:<34} frames {max(0,lo):>6}-{hi:<6} ({len(labs)} eventos)")

    rows = build_rows(os.path.expanduser(args.tracking), blocks, args.target)
    if args.exclude_block:
        before = len(rows)
        rows = [r for r in rows if r["block"] not in args.exclude_block]
        blocks = [b for b in blocks if b[0] not in args.exclude_block]
        print(f"excluidos {before - len(rows)} de bloques {args.exclude_block}")
    pos = sum(r["y"] for r in rows)
    print(f"\ntransiciones dentro de los bloques: {len(rows)}  "
          f"({pos} {args.target} / {len(rows)-pos} no-{args.target})")
    if pos < 20:
        print("⚠ muy pocos positivos: el resultado va a ser inestable.")

    cols = [c for c in feat.FIELDNAMES if c not in NOT_FEATURES]

    # ---- validacion leave-one-block-out ----
    print(f"\nVALIDACION leave-one-block-out (entrena en los otros bloques)")
    print(f"{'bloque de test':<34} {'n':>5} {'prec':>7} {'recall':>7} {'F1':>7}")
    print("-" * 64)
    tot = [0, 0, 0]
    thresholds = []
    probs = []
    for name, _lo, _hi, _labs in blocks:
        tr = [r for r in rows if r["block"] != name]
        te = [r for r in rows if r["block"] == name]
        if not te or not tr or sum(r["y"] for r in tr) < 5:
            print(f"{name:<34} {len(te):>5}   (sin datos suficientes)")
            continue
        Xtr, ytr = to_xy(tr, cols)
        Xte, yte = to_xy(te, cols)
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                             random_state=0)
        clf.fit(Xtr, ytr)
        # El umbral 0.5 es arbitrario y aca cuesta caro: con el, el modelo
        # queda mas preciso que las reglas pero MUCHO menos sensible (recall
        # 71% vs 86%), y pierde en F1. Se elige el umbral que maximiza F1
        # DENTRO del train -- nunca mirando el bloque de test, que seria
        # hacerse trampa al solitario.
        ptr = clf.predict_proba(Xtr)[:, 1]
        best_thr, best_f = 0.5, -1.0
        for k in range(5, 96):
            thr = k / 100.0
            tp_ = sum(1 for q, t in zip(ptr, ytr) if q >= thr and t == 1)
            fp_ = sum(1 for q, t in zip(ptr, ytr) if q >= thr and t == 0)
            fn_ = sum(1 for q, t in zip(ptr, ytr) if q < thr and t == 1)
            _p, _r, f_ = prf(tp_, fp_, fn_)
            if f_ > best_f:
                best_thr, best_f = thr, f_
        thresholds.append(best_thr)
        pte = clf.predict_proba(Xte)[:, 1]
        probs.extend(zip(pte, yte))
        pred = (pte >= best_thr).astype(int)
        tp = sum(1 for p, t in zip(pred, yte) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(pred, yte) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(pred, yte) if p == 0 and t == 1)
        p, r, f = prf(tp, fp, fn)
        tot[0] += tp; tot[1] += fp; tot[2] += fn
        print(f"{name:<34} {len(te):>5} {p:>6.1f}% {r:>6.1f}% {f:>6.1f}%")

    p, r, f = prf(*tot)
    print("-" * 64)
    print(f"{'GLOBAL':<34} {len(rows):>5} {p:>6.1f}% {r:>6.1f}% {f:>6.1f}%")
    if thresholds:
        print(f"umbrales elegidos en train: "
              f"{[f'{t:.2f}' for t in thresholds]}")

    # ---- ¿HAY SEÑAL? ----
    # Con 53% de positivos, el F1 engaña: decir "PASE" a todo ya da ~69%. Estas
    # dos metricas no se dejan enganiar por el desbalance.
    npos = sum(r["y"] for r in rows)
    tp_, fp_, fn_ = npos, len(rows) - npos, 0
    tp0, tr0, tf0 = prf(tp_, fp_, fn_)
    print(f"\nBASELINE TRIVIAL (todo es {args.target}):"
          f"   {tp0:.1f}% {tr0:.1f}% {tf0:.1f}%")

    if probs:
        pos = [q for q, t in probs if t == 1]
        neg = [q for q, t in probs if t == 0]
        if pos and neg:
            # AUC = probabilidad de rankear un positivo por encima de un
            # negativo al azar. 0.50 = no aprendio nada; 0.70+ = hay senal.
            wins = sum(1 for a in pos for b in neg if a > b)
            ties = sum(1 for a in pos for b in neg if a == b)
            auc = (wins + 0.5 * ties) / (len(pos) * len(neg))
            print(f"AUC (ranking, ciego al desbalance):  {auc:.3f}   "
                  f"{'sin senal' if auc < 0.6 else ('senal debil' if auc < 0.7 else 'HAY SENAL')}"
                  f"   [0.50 = azar]")
            # Precision con el recall fijado al de las reglas: comparacion justa
            # en el mismo punto de operacion.
            allq = sorted({q for q, _ in probs}, reverse=True)
            target_r = 0.863
            best = None
            for thr in allq:
                tp2 = sum(1 for q, t in probs if q >= thr and t == 1)
                fp2 = sum(1 for q, t in probs if q >= thr and t == 0)
                rec = tp2 / max(1, len(pos))
                if rec >= target_r:
                    best = 100 * tp2 / max(1, tp2 + fp2)
                    break
            if best is not None:
                print(f"precision del modelo al MISMO recall que las reglas "
                      f"(86.3%): {best:.1f}%  vs  59.5% de las reglas")

    # ---- baseline JUSTO: las reglas sobre las MISMAS transiciones ----
    if args.proposals:
        prop = []
        for r in csv.DictReader(open(os.path.expanduser(args.proposals))):
            if r["Type"] == args.target:
                prop.append(int(float(r["Start Frame"])))
        prop.sort()
        tp = fp = fn = 0
        for row in rows:
            lo = row["start_frame"] - BLOCK_PAD
            hi = row["end_frame"] + BLOCK_PAD
            rule = any(lo <= f <= hi for f in prop)
            if rule and row["y"] == 1:
                tp += 1
            elif rule:
                fp += 1
            elif row["y"] == 1:
                fn += 1
        rp, rr, rf = prf(tp, fp, fn)
        print(f"\nREGLAS sobre las MISMAS {len(rows)} transiciones:")
        print(f"{'':<34} {'':>5} {rp:>6.1f}% {rr:>6.1f}% {rf:>6.1f}%")
        print(f"\n-> el modelo {'GANA' if f > rf else 'PIERDE'} por "
              f"{abs(f - rf):.1f} puntos de F1")
    else:
        print("\n(pasa --proposals <csv> para comparar contra las reglas "
              "sobre las mismas transiciones)")

    # ---- modelo final sobre todo + importancias ----
    X, y = to_xy(rows, cols)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                         random_state=0).fit(X, y)
    try:
        from sklearn.inspection import permutation_importance
        imp = permutation_importance(clf, X, y, n_repeats=8, random_state=0)
        order = sorted(range(len(cols)), key=lambda i: -imp.importances_mean[i])
        print("\nFEATURES MAS IMPORTANTES")
        for i in order[:10]:
            print(f"  {cols[i]:<24} {imp.importances_mean[i]:.4f}")
    except Exception as exc:
        print(f"(sin importancias: {exc})")

    if args.out:
        import joblib
        joblib.dump({"model": clf, "columns": cols, "target": args.target},
                    os.path.expanduser(args.out))
        print(f"\nmodelo guardado: {args.out}")


if __name__ == "__main__":
    main()
