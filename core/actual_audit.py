"""Actual Model Audit: independently recomputes metrics from supplied predictions + ground truth.
No model is trained. Audits the supplied evaluation results only."""
import re
import numpy as np, pandas as pd
from sklearn import metrics as M

SEV = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
SCOPE = ("This mode audits the supplied evaluation results (predictions vs ground-truth labels). It cannot show that the test set was unseen, "
         "that no data leakage occurred, or that the original training process was valid.")

CLAIM_KEYS = [("balanced accuracy", "balanced_accuracy"), ("roc-auc", "roc_auc"), ("roc auc", "roc_auc"), ("pr-auc", "pr_auc"), ("pr auc", "pr_auc"),
              ("auc", "roc_auc"), ("accuracy", "accuracy"), ("precision", "precision"), ("recall", "recall"), ("f1", "f1"),
              ("r²", "r2"), ("r2", "r2"), ("r-squared", "r2"), ("rmse", "rmse"), ("mae", "mae"), ("mse", "mse")]

def parse_claim(text):
    """Return (metric, value, note). Only returns a metric when the claim is unambiguous: one metric, one number."""
    if not text or not text.strip(): return None, None, None
    t = text.lower(); found = []
    for kw, key in CLAIM_KEYS:
        if kw in t:
            found.append(key); t_masked = t.replace(kw, " " * len(kw))
            t = t_masked
    found = list(dict.fromkeys(found))
    nums = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%?)", text)
    nums = [(float(n), p) for n, p in nums if not re.fullmatch(r"[12]", n) or p or "." in n or True]
    if len(found) != 1: return None, None, "The claim names " + ("no recognised metric" if not found else "several metrics") + ", so it was not compared automatically. Name exactly one metric (e.g. 'accuracy')."
    if len(nums) != 1: return None, None, "The claim should contain exactly one number to be compared automatically."
    v, pct = nums[0]; key = found[0]
    if key in ("mae", "mse", "rmse", "r2"): return key, (v / 100 if pct and key == "r2" else v), None
    return key, (v / 100 if pct or v > 1 else v), None

def _bootstrap_ci(fn, y, p, n=500, seed=0):
    rs = np.random.RandomState(seed); L = len(y); vals = []
    for _ in range(n):
        i = rs.randint(0, L, L)
        try: vals.append(fn(y[i], p[i]))
        except Exception: pass
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) > 50 else None

def audit_predictions(df, pred_col, true_col, task, claim_text="", score_col=None, positive=None):
    if pred_col == true_col: raise ValueError("Prediction and ground-truth columns must be different.")
    for c in (pred_col, true_col):
        if c not in df.columns: raise ValueError(f"Column '{c}' was not found.")
    n_all = len(df); d = df[[pred_col, true_col] + ([score_col] if score_col else [])].copy()
    d = d.dropna(subset=[pred_col, true_col]); dropped = n_all - len(d)
    if len(d) == 0: raise ValueError("No rows remain after removing missing predictions/labels.")
    if len(d) < 10: raise ValueError("Fewer than 10 usable rows - too small to evaluate.")
    out = dict(mode="Actual Model Audit", task=task, n_rows=n_all, n_used=len(d), dropped=dropped, pred_col=pred_col, true_col=true_col, scope=SCOPE)
    F = []; out["findings"] = F
    def add(sev, tag, title, ev, why, rec): F.append(dict(severity=sev, tag=tag, title=title, evidence=ev, why=why, recommendation=rec))
    if dropped:
        pct = dropped / n_all * 100
        add("LOW" if pct < 5 else "MEDIUM", "missing", "Rows with missing predictions or labels", f"{dropped} of {n_all:,} rows ({pct:.1f}%) were excluded from metric calculation.",
            "If rows were dropped selectively, the reported metric may describe only part of the test set.", "Confirm that the reported number used the same rows.")
    m = {}; out["metrics"] = m
    claim_key, claim_val, claim_note = parse_claim(claim_text); out["claim"] = dict(text=claim_text, metric=claim_key, value=claim_val, note=claim_note)
    if task == "classification":
        yt = d[true_col].astype(str).str.strip().values; yp = d[pred_col].astype(str).str.strip().values
        classes = sorted(set(yt) | set(yp)); true_classes = sorted(set(yt))
        if len(true_classes) < 2: raise ValueError("Ground-truth column has only one class - metrics such as recall/AUC are not meaningful.")
        if len(classes) > 50 or len(true_classes) > 50: raise ValueError("Too many distinct labels (>50) - this looks like a regression target. Switch the task type.")
        unseen = sorted(set(yp) - set(yt))
        if unseen:
            add("MEDIUM", "labels", "Predicted labels absent from ground truth", f"Predictions contain label(s) never present in the ground truth: {unseen[:8]}.",
                "This often signals a label-encoding mismatch (e.g. 0/1 vs names), which can make every metric wrong.", "Verify both columns use the same label encoding.")
        vc = pd.Series(yt).value_counts(); out["class_dist"] = vc.to_dict()
        ratio = float(vc.iloc[0] / vc.iloc[-1]); out["imbalance_ratio"] = ratio
        binary = len(true_classes) == 2
        pos = str(positive) if positive is not None else str(vc.index[-1])
        if binary and pos not in true_classes: pos = str(vc.index[-1])
        out["positive"] = pos if binary else None
        m["accuracy"] = M.accuracy_score(yt, yp); m["balanced_accuracy"] = M.balanced_accuracy_score(yt, yp)
        kw = dict(zero_division=0, **(dict(average="binary", pos_label=pos) if binary else dict(average="macro")))
        m["precision"] = M.precision_score(yt, yp, **kw); m["recall"] = M.recall_score(yt, yp, **kw); m["f1"] = M.f1_score(yt, yp, **kw)
        out["averaging"] = f"positive class '{pos}'" if binary else "macro average"
        pr, rc, f1, sup = M.precision_recall_fscore_support(yt, yp, labels=classes, zero_division=0)
        out["per_class"] = {c: dict(precision=float(a), recall=float(b), f1=float(f), support=int(s)) for c, a, b, f, s in zip(classes, pr, rc, f1, sup)}
        out["confusion"] = M.confusion_matrix(yt, yp, labels=classes).tolist(); out["classes"] = classes
        m["majority_baseline"] = float(vc.iloc[0] / len(yt))
        out["score_note"] = None
        if score_col:
            try:
                s = pd.to_numeric(d[score_col], errors="coerce")
                if not binary: out["score_note"] = "ROC-AUC/PR-AUC not computed: a single score column is supported only for binary tasks."
                elif s.isna().any(): out["score_note"] = "ROC-AUC/PR-AUC not computed: the score column has missing or non-numeric values."
                else:
                    yb = (yt == pos).astype(int); m["roc_auc"] = M.roc_auc_score(yb, s); m["pr_auc"] = M.average_precision_score(yb, s)
                    out["score_note"] = f"Scores interpreted as higher = more likely '{pos}'."
            except Exception: out["score_note"] = "ROC-AUC/PR-AUC could not be computed from the score column."
        elif not binary: out["score_note"] = "ROC-AUC/PR-AUC not computed (multi-class or no probability column supplied)."
        else: out["score_note"] = "ROC-AUC/PR-AUC not computed: no probability/score column was selected."
        yb_t, yb_p = np.asarray(yt), np.asarray(yp)
        out["ci_accuracy"] = _bootstrap_ci(lambda a, b: float((a == b).mean()), yb_t, yb_p)
        # findings
        if len(yt) < 100:
            add("MEDIUM" if len(yt) < 50 else "LOW", "small", "Small evaluation set", f"Only {len(yt)} labelled rows" + (f"; accuracy 95% bootstrap interval {out['ci_accuracy'][0]*100:.1f}-{out['ci_accuracy'][1]*100:.1f}%." if out["ci_accuracy"] else "."),
                "Metrics from few rows have wide uncertainty.", "Evaluate on a larger held-out set or report confidence intervals.")
        if len(set(yp)) == 1:
            add("HIGH", "constant", "Model predicts a single class", f"Every prediction is '{yp[0]}'.", "The model is behaving like a constant predictor on this test set.", "Check the prediction pipeline and decision threshold.")
        if ratio >= 3:
            add("HIGH" if ratio >= 9 else "MEDIUM", "imbalance", "Imbalanced evaluation labels", "Ground-truth distribution: " + ", ".join(f"{k} {v/len(yt)*100:.1f}%" for k, v in vc.items()) + f" (ratio {ratio:.1f}:1).",
                "Accuracy can look high while minority classes are poorly handled.", "Report balanced accuracy, per-class recall and F1 alongside accuracy.")
            if m["accuracy"] - m["majority_baseline"] < 0.05:
                add("HIGH" if m["accuracy"] - m["majority_baseline"] < 0.02 else "MEDIUM", "baseline", "Accuracy close to majority-class rate",
                    f"Accuracy {m['accuracy']*100:.1f}% vs {m['majority_baseline']*100:.1f}% obtained by always predicting '{vc.index[0]}' ({(m['accuracy']-m['majority_baseline'])*100:+.1f} points).",
                    "Accuracy is only marginally better than a trivial predictor on these labels.", "Judge the model on minority-class metrics instead.")
        for c, v in out["per_class"].items():
            if v["support"] >= 5 and v["recall"] < 0.5 and (ratio >= 3 or v["recall"] < 0.3):
                add("HIGH", "recall", f"Weak recall for class '{c}'", f"Recall {v['recall']*100:.1f}% on {v['support']} examples (precision {v['precision']*100:.1f}%, F1 {v['f1']*100:.1f}%).",
                    "Most examples of this class are missed despite the overall metrics.", "Examine thresholds, class weights or more data for this class.")
        if m["accuracy"] - m["balanced_accuracy"] > 0.10:
            add("MEDIUM", "bal", "Large gap between accuracy and balanced accuracy", f"Accuracy {m['accuracy']*100:.1f}% vs balanced accuracy {m['balanced_accuracy']*100:.1f}%.", "Headline accuracy is driven by the larger classes.", "Lead with balanced accuracy.")
        if claim_key:
            if claim_key in m:
                diff = m[claim_key] - claim_val; ad = abs(diff) * 100
                ev = f"Claimed {claim_key.replace('_', ' ')} {claim_val*100:.1f}%; recomputed {m[claim_key]*100:.1f}% ({diff*100:+.1f} points; {out['averaging']} used for precision/recall/F1)." if claim_key in ("precision", "recall", "f1") else f"Claimed {claim_key.replace('_', ' ')} {claim_val*100:.1f}%; recomputed {m[claim_key]*100:.1f}% ({diff*100:+.1f} points)."
                out["claim"]["result"] = dict(claimed=claim_val, computed=m[claim_key], diff=diff)
                if ad <= 0.5: add("INFO", "claim", "Claim matches recomputed metric", ev, "The reported number is consistent with the supplied predictions.", "None - but see scope note below.")
                elif diff > 0: add("LOW", "claim", "Claim understates recomputed metric", ev, "Claimed value differs from the data supplied.", "Check which rows/averaging the claim used.")
                else: add("CRITICAL" if ad > 10 else "HIGH" if ad > 2 else "MEDIUM", "claim", "Claim not supported by supplied predictions", ev, "The reported performance is higher than the supplied predictions produce.", "Re-check the evaluation rows and reporting.")
            else: out["claim"]["note"] = f"The claimed metric ({claim_key.replace('_', ' ')}) could not be computed from the supplied columns."
    else:
        yt = pd.to_numeric(d[true_col], errors="coerce"); yp = pd.to_numeric(d[pred_col], errors="coerce"); ok = yt.notna() & yp.notna()
        bad = int((~ok).sum())
        if ok.sum() < 10: raise ValueError("Predictions and ground truth must be numeric for regression (too few numeric rows found).")
        if bad: add("MEDIUM", "labels", "Non-numeric values excluded", f"{bad} rows had non-numeric predictions or labels and were excluded.", "Metrics describe only the numeric rows.", "Clean the columns.")
        yt, yp = yt[ok].values, yp[ok].values; out["n_used"] = len(yt)
        if np.var(yt) == 0: raise ValueError("Ground truth is constant - R² is undefined.")
        mse = M.mean_squared_error(yt, yp)
        m.update(mae=M.mean_absolute_error(yt, yp), mse=mse, rmse=mse ** .5, r2=M.r2_score(yt, yp))
        base = float(np.mean(np.abs(yt - yt.mean()))); m["mean_baseline_mae"] = base
        res = yp - yt; out["residuals"] = res.tolist(); out["y_true"] = yt.tolist(); out["y_pred"] = yp.tolist()
        if m["r2"] < 0: add("HIGH", "r2", "Worse than predicting the mean", f"R² = {m['r2']:.3f}; MAE {m['mae']:.3f} vs {base:.3f} for a constant mean prediction.", "The predictions are less accurate than a trivial constant.", "Investigate the model or the column pairing.")
        elif m["mae"] > 0.9 * base: add("MEDIUM", "r2", "Little improvement over mean baseline", f"MAE {m['mae']:.3f} vs {base:.3f} for constant-mean prediction.", "Predictions add little over a trivial predictor.", "Improve features/model.")
        bias = res.mean() / (np.std(yt) or 1)
        if abs(bias) > 0.2: add("MEDIUM", "bias", "Systematic prediction bias", f"Mean residual {res.mean():+.3f} ({bias:+.2f} target std devs).", "Predictions are consistently " + ("high" if bias > 0 else "low") + ".", "Calibrate or correct the bias.")
        if m["mae"] > 0 and m["rmse"] / m["mae"] > 1.8: add("LOW", "tail", "Errors dominated by large outliers", f"RMSE {m['rmse']:.3f} is {m['rmse']/m['mae']:.1f}x MAE {m['mae']:.3f}.", "A few large errors drive the headline error.", "Inspect the worst predictions.")
        if len(yt) < 100: add("LOW", "small", "Small evaluation set", f"Only {len(yt)} rows.", "Metrics from few rows have wide uncertainty.", "Evaluate on more data.")
        out["ci_r2"] = _bootstrap_ci(lambda a, b: float(M.r2_score(a, b)), yt, yp)
        if claim_key:
            if claim_key in m and claim_key in ("mae", "mse", "rmse", "r2"):
                diff = m[claim_key] - claim_val; rel = abs(diff) / (abs(claim_val) or 1)
                lower_better = claim_key != "r2"; worse = diff > 0 if lower_better else diff < 0
                ev = f"Claimed {claim_key.upper()} {claim_val:g}; recomputed {m[claim_key]:.4g} ({diff:+.4g})."
                out["claim"]["result"] = dict(claimed=claim_val, computed=m[claim_key], diff=diff)
                if rel <= 0.02: add("INFO", "claim", "Claim matches recomputed metric", ev, "Consistent with the supplied predictions.", "None - but see scope note below.")
                elif worse: add("HIGH" if rel > 0.1 else "MEDIUM", "claim", "Claim not supported by supplied predictions", ev, "The data supplied gives worse performance than claimed.", "Re-check the evaluation rows and reporting.")
                else: add("LOW", "claim", "Claim understates recomputed metric", ev, "Claimed value differs from the data supplied.", "Check which rows the claim used.")
            else: out["claim"]["note"] = "That metric is not applicable to regression, so the claim was not compared."
    if out["claim"].get("note") and claim_text.strip(): add("INFO", "claim_note", "Claim not compared", out["claim"]["note"], "Only unambiguous claims are checked.", "Rephrase the claim with one metric and one number.")
    add("INFO", "scope", "Scope of this audit", SCOPE, "Correct predictions on a test set do not prove the evaluation setup was sound.", "Pair with a Dataset Benchmark Audit and a review of the training/evaluation split process.")
    F.sort(key=lambda f: -SEV.index(f["severity"]))
    c = {s: sum(f["severity"] == s for f in F) for s in SEV}
    out["status"] = ("HIGH-RISK EVALUATION" if c["CRITICAL"] else "SIGNIFICANT CONCERNS" if c["HIGH"] or c["MEDIUM"] >= 2 else "REVIEW RECOMMENDED" if c["MEDIUM"] or c["LOW"] >= 2 else "RELATIVELY SOUND")
    notable = [f["title"][0].lower() + f["title"][1:] for f in F if f["severity"] in ("MEDIUM", "HIGH", "CRITICAL")]
    out["conclusion"] = ("The supplied predictions show no major issues in the checks run here. " if not notable else "The supplied evaluation raises concerns: " + "; ".join(notable) + ". ") + "This does not verify how the model was trained or that the test set was unseen."
    return out
