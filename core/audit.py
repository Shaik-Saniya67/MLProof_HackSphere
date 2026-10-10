import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics as M

SUSPECT = ["outcome", "result", "target", "label", "post", "after", "final", "status", "churned", "approved", "leak"]
SEV = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

def detect_task(y):
    y = y.dropna()
    if y.dtype.kind in "fi" and y.nunique() > max(15, 0.05 * len(y)):
        return "regression"
    return "classification"

def profile(df, target):
    n = len(df); nun = df.nunique()
    num = df.select_dtypes("number").columns.tolist()
    return dict(rows=n, cols=df.shape[1], numeric=num, categorical=[c for c in df.columns if c not in num],
        missing_cells=int(df.isna().sum().sum()), missing_pct=float(df.isna().sum().sum() / max(df.size, 1) * 100),
        duplicates=int(df.duplicated().sum()),
        id_cols=[c for c in df.columns if c != target and n > 20 and nun[c] / n > 0.95 and df[c].dtype.kind in "iO"],
        constant=[c for c in df.columns if nun[c] <= 1],
        near_constant=[c for c in df.columns if nun[c] > 1 and df[c].value_counts(normalize=True, dropna=False).iloc[0] > 0.99])

def _pre(X):
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                        ("o", OneHotEncoder(handle_unknown="ignore", max_categories=15))]), cat)])

def _cls_metrics(m, Xte, yte, classes, minority):
    p = m.predict(Xte); r = {}
    r["accuracy"] = M.accuracy_score(yte, p); r["balanced_accuracy"] = M.balanced_accuracy_score(yte, p)
    avg = "binary" if len(classes) == 2 else "macro"
    kw = dict(average=avg, zero_division=0, **({"pos_label": minority} if avg == "binary" else {}))
    r["precision"] = M.precision_score(yte, p, **kw); r["recall"] = M.recall_score(yte, p, **kw); r["f1"] = M.f1_score(yte, p, **kw)
    r["per_class_recall"] = dict(zip(map(str, classes), M.recall_score(yte, p, labels=classes, average=None, zero_division=0)))
    r["minority_recall"] = r["per_class_recall"][str(minority)]
    r["confusion"] = M.confusion_matrix(yte, p, labels=classes).tolist()
    try:
        if hasattr(m, "predict_proba"):
            pr = m.predict_proba(Xte)
            if len(classes) == 2:
                pos = list(m.classes_).index(minority); yb = (yte == minority).astype(int)
                r["roc_auc"] = M.roc_auc_score(yb, pr[:, pos]); r["pr_auc"] = M.average_precision_score(yb, pr[:, pos])
            else:
                r["roc_auc"] = M.roc_auc_score(yte, pr, multi_class="ovr", labels=m.classes_)
    except Exception:
        pass
    return r

def run_audit(df, target, task=None):
    out = dict(findings=[], target=target)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    if len(df) < 30: raise ValueError("Dataset has fewer than 30 labelled rows - too small to audit reliably.")
    if len(df) > 50000:
        df = df.sample(50000, random_state=0).reset_index(drop=True); out["sampled"] = True
    task = task or detect_task(df[target]); out["task"] = task
    prof = profile(df, target); out["profile"] = prof
    F = out["findings"]
    def add(sev, tag, title, ev, why, rec): F.append(dict(severity=sev, tag=tag, title=title, evidence=ev, why=why, recommendation=rec))
    y = df[target]
    if y.nunique() < 2: raise ValueError("Target has only one class/value - nothing to evaluate.")
    if prof["missing_pct"] > 5:
        add("MEDIUM" if prof["missing_pct"] < 20 else "HIGH", "missing", "Substantial missing data",
            f"{prof['missing_cells']:,} missing cells ({prof['missing_pct']:.1f}% of all values).",
            "Imputation can hide bias and weaken learned patterns.", "Investigate why values are missing; collect or source more complete data.")
    if prof["duplicates"]:
        pct = prof["duplicates"] / len(df) * 100
        add("LOW" if pct < 1 else "MEDIUM", "duplicates", "Duplicate records",
            f"{prof['duplicates']} duplicate records detected ({pct:.1f}% of rows).", "Duplicates can fall on both sides of a split and inflate scores.", "Remove duplicates before splitting.")
    if prof["constant"] or prof["near_constant"]:
        add("INFO", "constant", "Constant / near-constant columns", f"Constant: {prof['constant']}; near-constant (>99% one value): {prof['near_constant']}.", "They carry almost no signal.", "Drop them.")
    if prof["id_cols"]:
        add("LOW", "id", "Identifier-like columns", f"Columns with near-unique values: {prof['id_cols']} (excluded from modelling).", "IDs can let models memorise rows.", "Keep IDs out of features.")
    drop = set(prof["id_cols"] + prof["constant"])
    X = df.drop(columns=[target] + list(drop))
    X = X.drop(columns=[c for c in X.columns if X[c].dtype == "O" and X[c].nunique() > 50])
    if X.shape[1] == 0: raise ValueError("No usable feature columns remain.")
    # leakage heuristics
    leak = []
    for c in X.columns:
        if any(k in c.lower() for k in SUSPECT): leak.append(f"'{c}' has a name suggesting post-outcome information")
    yc = pd.factorize(y)[0] if task == "classification" else y.astype(float)
    for c in X.select_dtypes("number").columns:
        try:
            r = abs(np.corrcoef(X[c].fillna(X[c].median()), yc)[0, 1])
            if r > 0.95: leak.append(f"'{c}' correlates {r:.2f} with the target")
        except Exception: pass
    if leak:
        add("HIGH", "leakage", "Potential target leakage detected", "; ".join(leak) + ".", "Features that encode the outcome make evaluation unrealistically optimistic. This is a heuristic, not confirmation.", "Check whether these features would exist at prediction time; remove if not.")
    # split
    strat = y if task == "classification" and y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)
    h = pd.util.hash_pandas_object(X.assign(_y=y), index=False)
    ov = int(h.loc[Xte.index].isin(set(h.loc[Xtr.index])).sum()); out["overlap"] = ov
    if ov:
        pct = ov / len(Xte) * 100
        add("HIGH" if pct > 2 else "MEDIUM", "contamination", "Potential train/test contamination", f"{ov} test records ({pct:.1f}%) also appear identically in the training set.", "Evaluation may be overly optimistic because the model has seen these rows.", "Deduplicate and recreate a clean split.")
    out["split"] = dict(train=len(Xtr), test=len(Xte))
    pre = _pre(X); res = {}; models = {}
    if task == "classification":
        vc = y.value_counts(); classes = sorted(vc.index.tolist(), key=str); minority = vc.index[-1]
        out["class_dist"] = {str(k): int(v) for k, v in vc.items()}
        ratio = vc.iloc[0] / vc.iloc[-1]; out["imbalance_ratio"] = float(ratio); out["minority"] = str(minority); out["minority_pct"] = float(vc.iloc[-1] / len(y) * 100)
        if ratio >= 3:
            add("HIGH" if ratio >= 9 else "MEDIUM", "imbalance", "Class imbalance", f"Class distribution: " + ", ".join(f"{k} {v/len(y)*100:.1f}%" for k, v in vc.items()) + f" (ratio {ratio:.1f}:1, {vc.iloc[-1]} minority examples).",
                f"Accuracy may be misleading because the minority class represents only {out['minority_pct']:.1f}% of the dataset.", "Report balanced accuracy, minority recall and PR-AUC; consider resampling or more minority data.")
        specs = {"Baseline (majority)": DummyClassifier(strategy="most_frequent"), "Logistic Regression": LogisticRegression(max_iter=1000), "Random Forest": RandomForestClassifier(150, random_state=0, n_jobs=-1)}
        for n, est in specs.items():
            m = Pipeline([("p", pre), ("m", est)]).fit(Xtr, ytr); models[n] = m
            res[n] = _cls_metrics(m, Xte, yte, classes, minority)
            res[n]["train_score"] = M.accuracy_score(ytr, m.predict(Xtr))
        main = "Random Forest"; key = "accuracy"
        cv = StratifiedKFold(5, shuffle=True, random_state=0) if strat is not None else KFold(5, shuffle=True, random_state=0)
        scoring = "accuracy"
    else:
        specs = {"Baseline (mean)": DummyRegressor(), "Linear Regression": LinearRegression(), "Random Forest": RandomForestRegressor(150, random_state=0, n_jobs=-1)}
        for n, est in specs.items():
            m = Pipeline([("p", pre), ("m", est)]).fit(Xtr, ytr); models[n] = m; p = m.predict(Xte)
            mse = M.mean_squared_error(yte, p)
            res[n] = dict(mae=M.mean_absolute_error(yte, p), mse=mse, rmse=mse ** .5, r2=M.r2_score(yte, p), train_score=m.score(Xtr, ytr))
        main = "Random Forest"; key = "r2"; cv = KFold(5, shuffle=True, random_state=0); scoring = "r2"
    out["models"] = res; out["main"] = main; r = res[main]; base = res[list(res)[0]]
    sc = cross_val_score(Pipeline([("p", pre), ("m", specs[main])]), X, y, cv=cv, scoring=scoring, n_jobs=1)
    out["cv"] = dict(scores=sc.tolist(), mean=float(sc.mean()), std=float(sc.std()), range=float(sc.max() - sc.min()), metric=scoring)
    if sc.std() > 0.03 or sc.max() - sc.min() > 0.08:
        add("MEDIUM", "cv", "Unstable cross-validation", f"{scoring} across 5 folds ranged {sc.min():.3f}-{sc.max():.3f} (std {sc.std():.3f}).", "Results depend heavily on which rows land in the test set.", "Use repeated/stratified CV and report the spread.")
    gap = r["train_score"] - r[key]; out["gap"] = float(gap)
    if gap > 0.10:
        add("HIGH" if gap > 0.2 else "MEDIUM", "overfit", "Possible overfitting", f"Training {key} {r['train_score']:.3f} vs test {r[key]:.3f} (gap {gap:.3f}).", "The model may be memorising the training data.", "Regularise, simplify the model or add data.")
    imp = r[key] - base[key]; out["improvement"] = float(imp)
    if imp < 0.05:
        add("MEDIUM" if imp > 0.01 else "HIGH", "baseline", "Little improvement over baseline", f"Model {key} {r[key]:.3f} vs baseline {base[key]:.3f} ({imp*100:+.1f} points).", "Performance is only marginally better than a trivial predictor.", "Try stronger features/models or collect better data.")
    if task == "classification" and "imbalance" in [f["tag"] for f in F] and r["minority_recall"] < 0.6:
        add("HIGH", "recall", "Weak minority-class performance", f"Recall on class '{minority}' is {r['minority_recall']*100:.1f}%; balanced accuracy {r['balanced_accuracy']:.3f}.", "Most minority cases are being missed despite headline accuracy.", "Optimise for minority recall/F1; use class weights or resampling.")
    if not F: add("INFO", "ok", "No major issues detected", "No heuristic check fired.", "Evidence is consistent with a sound evaluation (heuristics are not proof).", "Keep monitoring on fresh data.")
    F.sort(key=lambda f: -SEV.index(f["severity"]))
    c = {s: sum(f["severity"] == s for f in F) for s in SEV}
    out["status"] = ("HIGH-RISK EVALUATION" if c["CRITICAL"] else "SIGNIFICANT CONCERNS" if c["HIGH"] or c["MEDIUM"] >= 2 else
                     "REVIEW RECOMMENDED" if c["MEDIUM"] or c["LOW"] >= 2 else "RELATIVELY SOUND")
    out["claim_conclusion"] = conclude(out)
    out["_cm_classes"] = [str(c) for c in classes] if task == "classification" else None
    return out

PHRASE = {"imbalance": "the dataset is highly imbalanced", "recall": "minority-class recall is weak", "baseline": "the model provides only a small improvement over the baseline",
          "overfit": "training performance far exceeds test performance", "cv": "cross-validation results are unstable", "leakage": "potential target leakage was flagged",
          "contamination": "train/test records overlap", "missing": "a notable share of values is missing", "duplicates": "duplicate records exist"}
def conclude(o):
    tags = [f["tag"] for f in o["findings"] if f["tag"] in PHRASE]
    head = "The headline metric" + (" (accuracy)" if o["task"] == "classification" else " (R²)")
    if not tags: return f"{head} is consistent with the supporting evidence in this audit."
    ph = [PHRASE[t] for t in tags]
    s = ", ".join(ph[:-1]) + (", and " if len(ph) > 1 else "") + ph[-1]
    return f"{head} may overstate model quality because {s}."
