import numpy as np, pandas as pd
from sklearn.datasets import make_classification

def demo(kind):
    rs = np.random.RandomState(1)
    w = {"balanced": [.5, .5], "imbalanced": [.94, .06], "overfit": [.5, .5], "contaminated": [.9, .1]}[kind]
    n = 5000 if kind in ("imbalanced", "contaminated") else 800
    X, y = make_classification(n, 12 if kind != "overfit" else 40, n_informative=4 if kind == "imbalanced" else 6, n_redundant=0, weights=w,
                               flip_y=0.04 if kind != "balanced" else 0.01, class_sep=0.45 if kind == "imbalanced" else 1.0, random_state=1)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]); df["segment"] = rs.choice(list("ABC"), n)
    df.loc[rs.rand(n) < .08, "f1"] = np.nan; df["label"] = np.where(y == 1, "ClassB", "ClassA")
    if kind in ("imbalanced", "contaminated"):
        df = pd.concat([df, df.sample(40 if kind == "imbalanced" else 400, random_state=2)], ignore_index=True)
    if kind == "overfit": df["label"] = np.where(rs.rand(n) < .25, rs.choice(["ClassA", "ClassB"], n), df["label"])
    return df
