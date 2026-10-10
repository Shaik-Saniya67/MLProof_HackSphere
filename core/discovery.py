import numpy as np, pandas as pd
from sklearn import datasets as D

# Curated catalog: bundled scikit-learn datasets (real data, no network needed).
# Architecture: add a new loader to CATALOG (e.g. OpenML via fetch_openml) to extend.
CATALOG = {
    "Breast Cancer Wisconsin (Diagnostic)": dict(loader=D.load_breast_cancer, source="scikit-learn bundled (UCI ML Repository origin)", task="classification", domain="medical"),
    "Wine Recognition": dict(loader=D.load_wine, source="scikit-learn bundled (UCI ML Repository origin)", task="classification", domain="chemistry"),
    "Iris": dict(loader=D.load_iris, source="scikit-learn bundled", task="classification", domain="botany"),
    "Digits (8x8 images)": dict(loader=D.load_digits, source="scikit-learn bundled (UCI origin)", task="classification", domain="images"),
    "Diabetes progression": dict(loader=D.load_diabetes, source="scikit-learn bundled", task="regression", domain="medical"),
}

def load(name):
    b = CATALOG[name]["loader"](as_frame=True); df = b.frame.copy()
    return df, b.target.name

def characteristics(name):
    df, t = load(name); y = df[t]; info = dict(name=name, **{k: v for k, v in CATALOG[name].items() if k != "loader"})
    info.update(rows=len(df), features=df.shape[1] - 1, target=t, missing_pct=float(df.isna().mean().mean() * 100), duplicate_pct=float(df.duplicated().mean() * 100))
    if info["task"] == "classification":
        vc = y.value_counts(normalize=True); info.update(classes=int(y.nunique()), minority_pct=float(vc.min() * 100), ratio=float(vc.max() / vc.min()))
    return info

def needs_from(audit):
    n = []; tags = {f["tag"] for f in audit["findings"]}; p = audit["profile"]
    if "imbalance" in tags: n.append(f"better class balance than {audit['imbalance_ratio']:.1f}:1")
    if p["rows"] < 5000: n.append(f"more samples than {p['rows']:,}")
    if "missing" in tags: n.append(f"fewer missing values than {p['missing_pct']:.1f}%")
    if "duplicates" in tags: n.append("fewer duplicates")
    n.append("same target type (" + audit["task"] + ") and no obvious post-outcome features")
    return n

def recommend(audit):
    """Transparent rule-based matching: each candidate lists measurable reasons; no composite score."""
    out = []; p = audit["profile"]; cur_classes = len(audit.get("class_dist", {}))
    for name, meta in CATALOG.items():
        if meta["task"] != audit["task"]: continue
        c = characteristics(name); why = [f"✓ Same task type ({c['task']})"]; lim = ["Different domain/features from your data - may not transfer directly."]; rank = 0
        if c["task"] == "classification":
            if c["classes"] == cur_classes: why.append(f"✓ Same number of classes ({cur_classes})"); rank += 2
            else: lim.append(f"Has {c['classes']} classes vs {cur_classes} in yours.")
            if "imbalance" in {f['tag'] for f in audit["findings"]}:
                if c["ratio"] < audit["imbalance_ratio"]: why.append(f"✓ Better balance ({c['ratio']:.1f}:1 vs {audit['imbalance_ratio']:.1f}:1)"); rank += 2
                else: lim.append("Not better balanced than your data.")
        if c["rows"] > p["rows"]: why.append(f"✓ {c['rows']/p['rows']:.1f}× more records"); rank += 1
        else: lim.append(f"Smaller ({c['rows']:,} rows vs {p['rows']:,}).")
        if c["missing_pct"] < p["missing_pct"]: why.append(f"✓ Lower missing-data rate ({c['missing_pct']:.1f}% vs {p['missing_pct']:.1f}%)"); rank += 1
        if c["duplicate_pct"] == 0: why.append("✓ No duplicate rows"); rank += 1
        out.append(dict(info=c, why=why, limits=lim, rank=rank))
    return sorted(out, key=lambda r: -r["rank"])
