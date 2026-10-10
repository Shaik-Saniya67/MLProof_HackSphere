"""Model 1 workflow logic: load -> analyze -> (optional) clean -> train -> compare.

Pure pandas / scikit-learn; no Streamlit imports, so everything here can be tested on its own.
Leakage rule: every statistic-based step (imputation, scaling, encoding) lives inside a scikit-learn
Pipeline that is fitted on the training split only. The cleaning step only performs operations that do
not learn anything from the data's values or labels (dedupe, drop constant/ID/sparse columns, drop rows).
"""
import re, time, warnings
import numpy as np, pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_datetime64_any_dtype
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

MAX_TRAIN_ROWS = 50_000
MIN_ROWS = 30
CLASS_MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "CatBoost", "TabNet", "Dummy baseline"]
REG_MODELS = ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost", "CatBoost", "TabNet", "Dummy baseline"]
OPTIONAL_PKG = {"XGBoost": ("xgboost", "pip install xgboost"), "CatBoost": ("catboost", "pip install catboost"),
                "TabNet": ("pytorch_tabnet", "pip install torch pytorch-tabnet  (see requirements-optional.txt)")}
DEFAULT_OFF = {"TabNet"}  # slow deep-learning model: opt-in


def model_availability():
    """{model name: None if usable, else a short reason}. Checks installation only; nothing heavy is imported."""
    import importlib.util
    out = {}
    for name, (pkg, hint) in OPTIONAL_PKG.items():
        try:
            ok = importlib.util.find_spec(pkg) is not None and (pkg != "pytorch_tabnet" or importlib.util.find_spec("torch") is not None)
        except Exception:
            ok = False
        out[name] = None if ok else f"not installed - {hint}"
    return out
ID_NAME = re.compile(r"(^|[_\s-])(id|uuid|guid|index|idx)$|^(id|uuid|guid)([_\s-]|$)", re.I)
LEAK_NAME = re.compile(r"(outcome|result|target|label|after|post_|_post|final|leak)", re.I)


# ----------------------------------------------------------------------------- loading
def read_csv_safe(file):
    """Read an uploaded CSV. Raises ValueError with a user-friendly message."""
    last = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            break
        except UnicodeDecodeError as e:
            last = e
        except pd.errors.EmptyDataError:
            raise ValueError("The file is empty.")
        except Exception:
            raise ValueError("This file could not be read as a CSV. Check that it is a comma-separated text file.")
    else:
        raise ValueError("The file's text encoding is not supported.")
    if df.shape[1] < 2:
        raise ValueError("The CSV needs at least two columns (features and a target). Check the delimiter.")
    if len(df) == 0:
        raise ValueError("The CSV has a header but no rows.")
    df.columns = [str(c).strip() or f"column_{i}" for i, c in enumerate(df.columns)]
    return df


def column_overview(df):
    n = max(len(df), 1)
    return pd.DataFrame({"Column": df.columns, "Type": [str(t) for t in df.dtypes],
                         "Non-null": df.notna().sum().values, "Missing": df.isna().sum().values,
                         "Missing %": (df.isna().mean().values * 100).round(1), "Unique": df.nunique().values})


def make_sample(kind):
    """Messy, synthetic example datasets (for demos and tests)."""
    rs = np.random.RandomState(11 if kind == "classification" else 12)
    n = 900
    age = rs.normal(42, 12, n).clip(18, 80).round()
    income = rs.lognormal(10.3, .45, n).round()
    tenure = rs.randint(0, 120, n).astype(float)
    plan = rs.choice(["basic", "plus", "pro"], n, p=[.55, .3, .15])
    region = rs.choice(["north", "south", "east", "west"], n)
    df = pd.DataFrame({"customer_id": np.arange(10001, 10001 + n), "age": age, "income": income, "tenure_months": tenure,
                       "plan": plan, "region": region, "country": "IN"})
    if kind == "classification":
        z = (60 - tenure) / 18 + (plan == "basic") * 1.2 + (income < 24000) * 1.6 + rs.randn(n) * .7
        df["churned"] = np.where(z > np.quantile(z, .88), "yes", "no")
    else:
        df["price"] = (20000 + income * .8 + (age - 40) * 150 + (plan == "pro") * 9000 + tenure * 60 + rs.randn(n) * 3000).round()
        df.loc[rs.choice(n, 5, replace=False), "price"] += 80000  # outliers
    for c, k in (("age", 60), ("income", 45), ("region", 50)):
        df.loc[rs.choice(n, k, replace=False), c] = np.nan
    df.loc[rs.choice(n, 20, replace=False), "plan"] = np.nan
    df = pd.concat([df, df.sample(25, random_state=3)], ignore_index=True)
    return df


# ----------------------------------------------------------------------------- analysis
def _feature_cols(df, target):
    return [c for c in df.columns if c != target]


def _integer_valued(s):
    v = s.dropna()
    return len(v) > 0 and bool(np.all(np.mod(v.values.astype(float), 1) == 0))


def id_like_columns(df, target=None):
    """Columns that look like row identifiers. Returns {column: reason}."""
    out, n = {}, len(df)
    for c in _feature_cols(df, target):
        s = df[c]
        if is_datetime64_any_dtype(s):
            continue
        reasons = []
        named = bool(ID_NAME.search(c))
        if named:
            reasons.append("name looks like an identifier")
        if n >= 20 and not is_bool_dtype(s):
            ratio = s.nunique(dropna=True) / n
            if ratio >= 0.98:
                if not is_numeric_dtype(s):
                    reasons.append(f"{ratio:.0%} of values are unique text")
                elif _integer_valued(s) and (named or s.dropna().is_monotonic_increasing or s.dropna().is_monotonic_decreasing):
                    reasons.append(f"{ratio:.0%} of values are unique whole numbers in sequence")
        if reasons:
            out[c] = "; ".join(reasons)
    return out


def analyze(df, target=None):
    n = len(df)
    feats = _feature_cols(df, target)
    miss = df.isna().sum()
    miss_df = pd.DataFrame({"Column": miss.index, "Missing": miss.values, "Missing %": (miss.values / max(n, 1) * 100).round(1)})
    miss_df = miss_df[miss_df.Missing > 0].sort_values("Missing", ascending=False).reset_index(drop=True)
    const = [c for c in df.columns if df[c].nunique(dropna=True) <= 1 and c != target]
    num = [c for c in df.columns if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])]
    dt = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
    cat = [c for c in df.columns if c not in num and c not in dt]
    outl = []
    for c in num:
        s = df[c].dropna()
        if len(s) < 8 or s.nunique() <= 2:
            continue
        q1, q3 = s.quantile(.25), s.quantile(.75); iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        k = int(((s < lo) | (s > hi)).sum())
        if k:
            outl.append(dict(Column=c, Outliers=k, **{"Outlier %": round(k / len(s) * 100, 1)}, **{"Lower fence": lo, "Upper fence": hi, "Min": s.min(), "Max": s.max()}))
    res = dict(rows=n, cols=df.shape[1], missing=miss_df, missing_cells=int(miss.sum()), rows_with_missing=int(df.isna().any(axis=1).sum()),
               duplicates=int(df.duplicated().sum()), constant=const, id_like=id_like_columns(df, target), numeric=num, categorical=cat, datetime=dt,
               outliers=pd.DataFrame(outl), high_missing=[c for c in df.columns if df[c].isna().mean() > .4 and c != target])
    if target and target in df.columns:
        res["target_missing"] = int(df[target].isna().sum())
    return res


def suggest_task(y):
    """Return (task, plain-language reason)."""
    s = y.dropna()
    nun = s.nunique()
    if len(s) == 0:
        return "classification", "The target has no values."
    if is_bool_dtype(s) or not is_numeric_dtype(s):
        return "classification", f"The target is text/categories with {nun} distinct values."
    if nun == 2:
        return "classification", "The target has exactly 2 distinct values (binary)."
    is_int = bool(np.all(np.mod(s.values.astype(float), 1) == 0))
    if is_int and nun <= 15:
        return "classification", f"The target holds whole numbers with only {nun} distinct values, so they look like class labels."
    if not is_int and nun <= 8 and nun / len(s) < .05:
        return "classification", f"The target has only {nun} distinct numeric values."
    return "regression", f"The target is numeric with {nun} distinct values, so it looks like a continuous quantity."


def validate_target(df, target, task):
    """Return (errors, warnings, info). Errors block training."""
    err, warn, info = [], [], {}
    if target not in df.columns:
        return [f"The target column '{target}' is not in this dataset (it may have been removed during cleaning)."], warn, info
    y = df[target]
    nmiss = int(y.isna().sum()); info["target_missing"] = nmiss
    yv = y.dropna()
    if nmiss:
        warn.append(f"{nmiss} row(s) have no target value and will be left out of training (they are not changed in your data).")
    if len(yv) < MIN_ROWS:
        err.append(f"Only {len(yv)} rows have a target value; at least {MIN_ROWS} are needed for a train/test comparison.")
        return err, warn, info
    if task == "classification":
        vc = yv.value_counts()
        info["n_classes"] = len(vc)
        if len(vc) < 2:
            err.append("The target has only one class, so there is nothing to classify.")
        elif len(vc) > 30:
            err.append(f"The target has {len(vc)} distinct classes, which is too many for this tool. If it is a number, try regression.")
        else:
            small = vc[vc < 2]
            if len(small):
                warn.append(f"{len(small)} class(es) have only one example ({', '.join(map(str, small.index[:5]))}). Results for them are unreliable and the split cannot be stratified.")
            elif vc.min() < 10:
                warn.append(f"The smallest class has only {int(vc.min())} examples, so metrics for it will vary a lot.")
    else:
        if not is_numeric_dtype(yv) or is_bool_dtype(yv):
            err.append("Regression needs a numeric target, but this column is not numeric. Choose classification or another target.")
        elif yv.nunique() < 2:
            err.append("The target has the same value in every row, so there is nothing to predict.")
        elif yv.nunique() < 6:
            warn.append(f"The target has only {yv.nunique()} distinct values; classification may suit it better.")
    if target in id_like_columns(df[[target]]):
        warn.append("This column looks like a row identifier, not something worth predicting.")
    return err, warn, info


# ----------------------------------------------------------------------------- cleaning
def cleaning_preview_counts(df, target=None):
    a = analyze(df, target)
    return dict(rows=a["rows"], cols=a["cols"], missing_cells=a["missing_cells"], duplicates=a["duplicates"], constant=len(a["constant"]))


def apply_cleaning(raw, opts, target=None):
    """Return (cleaned_df, log). The raw DataFrame is never modified. The target column is never dropped.

    opts keys: dedupe(bool), drop_constant(list), drop_id(list), sparse_threshold(None|float %),
               row_missing('keep'|'target'|'any')
    """
    df = raw.copy()
    log = []

    def note(action, detail, rows=0, cols=0):
        log.append(dict(Action=action, Detail=detail, **{"Rows removed": rows, "Columns removed": cols}))

    cols_drop = []
    for key, label in (("drop_constant", "Removed constant column"), ("drop_id", "Removed ID-like column")):
        for c in opts.get(key, []) or []:
            if c in df.columns and c != target and c not in cols_drop:
                cols_drop.append(c); note(label, c, cols=1)
    thr = opts.get("sparse_threshold")
    if thr:
        for c in list(df.columns):
            if c != target and c not in cols_drop and df[c].isna().mean() * 100 > thr:
                cols_drop.append(c); note("Removed sparse column", f"{c} ({df[c].isna().mean():.0%} missing, limit {thr:.0f}%)", cols=1)
    df = df.drop(columns=cols_drop)
    mode = opts.get("row_missing", "keep")
    if mode == "target" and target in df.columns:
        k = int(df[target].isna().sum())
        if k:
            df = df[df[target].notna()]; note("Removed rows with missing target", f"{k} row(s)", rows=k)
    elif mode == "any":
        k = int(df.isna().any(axis=1).sum())
        if k:
            df = df.dropna(); note("Removed rows with any missing value", f"{k} row(s)", rows=k)
    if opts.get("dedupe"):
        k = int(df.duplicated().sum())
        if k:
            df = df.drop_duplicates(keep="first"); note("Removed exact duplicate rows", f"{k} row(s), first occurrence kept", rows=k)
    df = df.reset_index(drop=True)
    if not log:
        note("No changes", "No cleaning action changed the data.")
    return df, log


# ----------------------------------------------------------------------------- features
def feature_report(df, target, feats=None):
    """Decide which columns can be used as features. Returns (usable, excluded{col: reason}, leak_suspects{col: reason})."""
    n = len(df)
    ids = id_like_columns(df, target)
    usable, excluded, leaks = [], {}, {}
    y = df[target] if target in df.columns else None
    for c in (feats if feats is not None else _feature_cols(df, target)):
        if c == target or c not in df.columns:
            continue
        s = df[c]
        if c in ids:
            excluded[c] = "ID-like column (" + ids[c] + ")"
        elif s.nunique(dropna=True) <= 1:
            excluded[c] = "constant column"
        elif is_datetime64_any_dtype(s):
            excluded[c] = "date/time column (not supported)"
        elif not is_numeric_dtype(s) and s.nunique() > max(50, 0.5 * n):
            excluded[c] = "free text or very high-cardinality column"
        else:
            usable.append(c)
            if LEAK_NAME.search(c):
                leaks[c] = "name suggests it may be derived from the outcome"
            if y is not None and is_numeric_dtype(s) and is_numeric_dtype(y) and not is_bool_dtype(s):
                m = s.notna() & y.notna()
                if m.sum() > 10 and s[m].nunique() > 1 and y[m].nunique() > 1:
                    r = abs(np.corrcoef(s[m].astype(float), y[m].astype(float))[0, 1])
                    if r > 0.98:
                        leaks[c] = f"correlates {r:.3f} with the target"
    return usable, excluded, leaks


def _to_model_frame(df, cols):
    X = df[cols].copy()
    for c in cols:
        s = X[c]
        if is_bool_dtype(s) or not is_numeric_dtype(s):
            X[c] = s.map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
    return X


def build_preprocessor(X, impute_num="median", impute_cat="most_frequent"):
    num = [c for c in X.columns if is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    ni = SimpleImputer(strategy=impute_num)
    ci = SimpleImputer(strategy="constant", fill_value="Missing") if impute_cat == "constant" else SimpleImputer(strategy="most_frequent")
    ohe = OneHotEncoder(handle_unknown="ignore", max_categories=15, sparse_output=False)
    parts = []
    if num:
        parts.append(("num", Pipeline([("imp", ni), ("sc", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([("imp", ci), ("ohe", ohe)]), cat))
    return ColumnTransformer(parts)


# ----------------------------------------------------------------------------- training
class _LabelEncodedModel(BaseEstimator):
    """Small sklearn-style wrapper so boosting/deep models that need numeric labels work with the text class labels
    used everywhere else. The encoder is fitted on the training labels only (inside the Pipeline)."""
    def __init__(self, build, classifier=True, scale_y=False):
        self.build, self.classifier, self.scale_y = build, classifier, scale_y

    def __sklearn_is_fitted__(self):
        return hasattr(self, "m_")

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        if self.classifier:
            self.classes_ = np.array(sorted(set(y)))
            lut = {c: i for i, c in enumerate(self.classes_)}
            y = np.array([lut[v] for v in y])
        else:
            y = np.asarray(y, dtype=np.float64)
            self.mu_, self.sd_ = (float(y.mean()), float(y.std()) or 1.0) if self.scale_y else (0.0, 1.0)  # training rows only
            y = ((y - self.mu_) / self.sd_).astype(np.float32)
        self.m_ = self.build(X, y, len(getattr(self, "classes_", [])))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        pred = np.asarray(self.m_.predict(X)).reshape(len(X), -1)[:, 0]
        return self.classes_[pred.astype(int)] if self.classifier else pred.astype(float) * self.sd_ + self.mu_


def _xgb_builder(task, p):
    def build(X, y, n_classes):
        from xgboost import XGBClassifier, XGBRegressor
        kw = dict(n_estimators=p["xgb_trees"], max_depth=p["xgb_depth"], learning_rate=0.1, random_state=p["seed"], n_jobs=1, verbosity=0)
        m = XGBClassifier(**kw, eval_metric="logloss") if task == "classification" else XGBRegressor(**kw)
        return m.fit(X, y)
    return build


def _cat_builder(task, p):
    def build(X, y, n_classes):
        from catboost import CatBoostClassifier, CatBoostRegressor
        kw = dict(iterations=p["cat_iters"], depth=p["cat_depth"], learning_rate=0.1, random_seed=p["seed"], verbose=0, allow_writing_files=False, thread_count=1)
        m = CatBoostClassifier(**kw) if task == "classification" else CatBoostRegressor(**kw)
        return m.fit(X, y)
    return build


def _tabnet_builder(task, p):
    def build(X, y, n_classes):
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
        torch.manual_seed(p["seed"]); np.random.seed(p["seed"])
        kw = dict(seed=p["seed"], verbose=0, device_name="cpu", n_d=8, n_a=8, n_steps=3)
        # early stopping uses a validation slice carved from the TRAINING rows only - never the test set
        use_val = len(X) >= 300
        if use_val:
            Xa, Xv, ya, yv = train_test_split(X, y, test_size=0.15, random_state=p["seed"], stratify=y if task == "classification" and np.bincount(y).min() >= 2 else None)
        else:
            Xa, ya = X, y
        if task == "classification":
            m = TabNetClassifier(**kw)
        else:
            m = TabNetRegressor(**kw); ya = ya.reshape(-1, 1)
            if use_val:
                yv = yv.reshape(-1, 1)
        fit_kw = dict(max_epochs=p["tabnet_epochs"], batch_size=min(256, len(Xa)), virtual_batch_size=min(128, len(Xa)), drop_last=False)
        if use_val:
            fit_kw.update(eval_set=[(Xv, yv)], patience=8)
        m.fit(Xa, ya, **fit_kw)
        return m
    return build


def make_models(task, names, p):
    seed = p["seed"]
    if task == "classification":
        d = {"Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
             "Decision Tree": DecisionTreeClassifier(max_depth=p["tree_depth"], random_state=seed),
             "Random Forest": RandomForestClassifier(n_estimators=p["rf_trees"], max_depth=p["rf_depth"], random_state=seed, n_jobs=1),
             "Dummy baseline": DummyClassifier(strategy="most_frequent")}
    else:
        d = {"Linear Regression": LinearRegression(),
             "Decision Tree": DecisionTreeRegressor(max_depth=p["tree_depth"], random_state=seed),
             "Random Forest": RandomForestRegressor(n_estimators=p["rf_trees"], max_depth=p["rf_depth"], random_state=seed, n_jobs=1),
             "Dummy baseline": DummyRegressor(strategy="mean")}
    clf = task == "classification"
    for n, b in (("XGBoost", _xgb_builder), ("CatBoost", _cat_builder), ("TabNet", _tabnet_builder)):
        d[n] = _LabelEncodedModel(b(task, p), classifier=clf, scale_y=(n == "TabNet"))
    return {n: d[n] for n in names if n in d}


def default_params():
    return dict(seed=42, test_size=0.25, tree_depth=5, rf_trees=100, rf_depth=None, impute_num="median", impute_cat="most_frequent",
                xgb_trees=200, xgb_depth=4, cat_iters=200, cat_depth=6, tabnet_epochs=30)


def train_and_compare(df, target, task, features, model_names, params, positive=None, progress=None):
    """Fit every model on the same training split; evaluate on the held-out test split.
    Preprocessing is part of each Pipeline, so it is fitted on training data only."""
    p = {**default_params(), **params}
    err, warn, _ = validate_target(df, target, task)
    if err:
        raise ValueError(err[0])
    if not features:
        raise ValueError("Select at least one feature column.")
    data = df[df[target].notna()]
    notes = list(warn)
    if len(data) > MAX_TRAIN_ROWS:
        data = data.sample(MAX_TRAIN_ROWS, random_state=p["seed"]); notes.append(f"Large dataset: a random sample of {MAX_TRAIN_ROWS:,} rows was used for training and testing.")
    X = _to_model_frame(data, features)
    y = data[target]
    if task == "classification":
        y = y.map(str)
    else:
        y = y.astype(float)
    strat = None
    if task == "classification":
        vc = y.value_counts()
        if vc.min() >= 2 and int(len(y) * p["test_size"]) >= len(vc):
            strat = y
        else:
            notes.append("The split could not be stratified (a class is too small), so class proportions may differ between train and test.")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=p["test_size"], random_state=p["seed"], stratify=strat)
    out = dict(task=task, target=target, features=list(features), n_train=len(Xtr), n_test=len(Xte), notes=notes, params=p, results={}, failed={})
    classes = sorted(y.unique()) if task == "classification" else None
    if task == "classification":
        vc = y.value_counts(); minority = str(positive) if positive is not None and str(positive) in vc.index else str(vc.index[-1])
        out.update(classes=classes, positive=minority if len(classes) == 2 else None, averaging=f"positive class '{minority}'" if len(classes) == 2 else "macro average over classes",
                   class_dist=vc.to_dict(), test_dist=yte.value_counts().to_dict(), majority_share=float(vc.iloc[0] / len(y)), minority_share=float(vc.iloc[-1] / len(y)),
                   imbalance_ratio=float(vc.iloc[0] / vc.iloc[-1]), minority=str(vc.index[-1]))
    avail = model_availability()
    models = make_models(task, model_names, p)
    for n in model_names:
        if avail.get(n):
            out["failed"][n] = "Skipped - " + avail[n]
            models.pop(n, None)
    for i, (name, est) in enumerate(models.items()):
        if progress:
            progress(i / len(models), f"Training {name}…")
        pipe = Pipeline([("prep", build_preprocessor(Xtr, p["impute_num"], p["impute_cat"])), ("model", est)])
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(Xtr, ytr)
                pred = pipe.predict(Xte)
            if task == "classification":
                r = _cls_metrics(yte.values, pred, classes, out["positive"])
            else:
                mse = mean_squared_error(yte, pred)
                r = dict(MAE=float(mean_absolute_error(yte, pred)), RMSE=float(mse ** .5), R2=float(r2_score(yte, pred)),
                         y_true=yte.tolist(), y_pred=[float(v) for v in pred])
            r["train_seconds"] = round(time.time() - t0, 2)
            out["results"][name] = r
        except Exception as e:  # keep the other models going
            out["failed"][name] = f"{type(e).__name__}: {str(e)[:160]}"
    if progress:
        progress(1.0, "Done")
    if not out["results"]:
        raise ValueError("No model could be trained on this data. " + "; ".join(out["failed"].values())[:200])
    return out


def _cls_metrics(y, pred, classes, positive):
    avg = dict(average="binary", pos_label=positive) if positive is not None else dict(average="macro")
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, zero_division=0, **avg)
    p_c, r_c, f_c, sup = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    return dict(Accuracy=float(accuracy_score(y, pred)), Precision=float(pr), Recall=float(rc), F1=float(f1),
                **{"Balanced accuracy": float(balanced_accuracy_score(y, pred))},
                confusion=confusion_matrix(y, pred, labels=classes).tolist(),
                per_class=[dict(Class=c, Precision=float(a), Recall=float(b), F1=float(f), Support=int(s)) for c, a, b, f, s in zip(classes, p_c, r_c, f_c, sup)])


# ----------------------------------------------------------------------------- comparison
def results_table(out):
    rows = []
    cols = ["Accuracy", "Balanced accuracy", "Precision", "Recall", "F1"] if out["task"] == "classification" else ["MAE", "RMSE", "R2"]
    for n, r in out["results"].items():
        rows.append({"Model": n, **{c: round(r[c], 4) for c in cols}, "Train time (s)": r["train_seconds"]})
    return pd.DataFrame(rows)


def best_model(out):
    """Best non-baseline model by F1 (classification) or RMSE (regression)."""
    cand = {n: r for n, r in out["results"].items() if n != "Dummy baseline"} or out["results"]
    if out["task"] == "classification":
        return max(cand, key=lambda n: cand[n]["F1"]), "F1"
    return min(cand, key=lambda n: cand[n]["RMSE"]), "RMSE"


def explain(out):
    """Return (summary_sentences, warnings)."""
    res, task = out["results"], out["task"]
    best, crit = best_model(out)
    msgs, warns = [], []
    b = res[best]
    if task == "classification":
        msgs.append(f"{best} has the highest {crit} among the trained models ({b['F1']:.3f}), with accuracy {b['Accuracy']:.1%} on {out['n_test']:,} held-out test rows.")
        base = res.get("Dummy baseline")
        if base:
            diff = (b["Accuracy"] - base["Accuracy"]) * 100
            msgs.append(f"The dummy baseline (always predicting the most common class) scores {base['Accuracy']:.1%} accuracy, so {best} is {abs(diff):.1f} percentage points {'above' if diff >= 0 else 'below'} it.")
        if out["imbalance_ratio"] >= 3:
            warns.append(f"Accuracy may be misleading: the largest class is {out['majority_share']:.0%} of the data (imbalance {out['imbalance_ratio']:.1f}:1). "
                         f"A model can score high accuracy while missing the rare class, so compare Recall, F1 and Balanced accuracy as well.")
            if "Dummy baseline" in res and res["Dummy baseline"]["Accuracy"] >= .85 and b["Recall"] < .5:
                warns.append(f"{best} finds only {b['Recall']:.0%} of the rare cases (recall) even though its accuracy is {b['Accuracy']:.1%}.")
    else:
        msgs.append(f"{best} has the lowest {crit} ({b['RMSE']:.4g}); MAE is {b['MAE']:.4g} and R² is {b['R2']:.3f} on {out['n_test']:,} held-out test rows.")
        base = res.get("Dummy baseline")
        if base:
            msgs.append(f"The mean-prediction baseline has RMSE {base['RMSE']:.4g}; {best} is {(1 - b['RMSE'] / base['RMSE']) * 100:.0f}% lower." if base["RMSE"] > 0 else "")
        if b["R2"] < 0.1:
            warns.append("R² is close to zero or negative: the model explains little more than the average value.")
    msgs.append("These are results from one train/test split of your data. They describe this evaluation, not a guarantee of how the model will perform on future data.")
    return [m for m in msgs if m], warns


def summary_dict(name, raw_analysis, clean_log, clean_after, use_label, out, impute_note=""):
    """Everything the report needs, as plain data."""
    d = dict(dataset=name, trained_on=use_label, findings=dict(
        rows=raw_analysis["rows"], columns=raw_analysis["cols"], missing_cells=raw_analysis["missing_cells"], rows_with_missing=raw_analysis["rows_with_missing"],
        duplicate_rows=raw_analysis["duplicates"], constant_columns=raw_analysis["constant"], id_like_columns=raw_analysis["id_like"],
        numeric_columns=len(raw_analysis["numeric"]), categorical_columns=len(raw_analysis["categorical"]),
        columns_with_outliers=[] if raw_analysis["outliers"].empty else raw_analysis["outliers"]["Column"].tolist()),
        cleaning=clean_log or [], cleaned_shape=clean_after)
    if out:
        best, crit = best_model(out)
        d["model"] = dict(task=out["task"], target=out["target"], n_features=len(out["features"]), features=out["features"], train_rows=out["n_train"], test_rows=out["n_test"],
                          settings={k: v for k, v in out["params"].items()}, models_trained=list(out["results"]), models_failed=out["failed"], averaging=out.get("averaging"),
                          best_model=best, best_by=crit, notes=out["notes"])
        d["metrics"] = results_table(out).to_dict(orient="records")
        d["explanation"], d["warnings"] = explain(out)
    return d
