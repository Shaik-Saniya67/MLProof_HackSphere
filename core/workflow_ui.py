"""Model 1 guided workflow UI: Upload -> Analyze -> Clean -> Train -> Compare -> Export.

State lives in st.session_state["wf"] (plain data) so it survives page changes. Navigation between steps
uses a 'pending step' that is applied BEFORE the step selector is created, so we never modify a widget's
state after it exists (the Streamlit error we hit earlier).
"""
from html import escape as esc
import pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from core import workflow as W
from reports import workflow_report as R

STEPS = ["1 · Upload", "2 · Analyze", "3 · Clean", "4 · Train", "5 · Compare", "6 · Export"]
RED, INK, AMBER, TEAL = "#D8232A", "#1B120F", "#C98A1B", "#1F6F6B"
PAL = [RED, INK, AMBER, TEAL]
MODEL_COL = {"Logistic Regression": RED, "Linear Regression": RED, "Decision Tree": INK, "Random Forest": AMBER, "XGBoost": TEAL,
             "CatBoost": "#7A5C8E", "TabNet": "#4F78B0", "Dummy baseline": "#A39A89"}
CSS = """<style>
.st-key-wf_step_radio [role=radiogroup]{gap:.4rem;flex-wrap:wrap}
.st-key-wf_step_radio label{border:1px solid #CDBFA5;border-radius:99px;padding:.15rem .9rem;background:#FFFDF8;cursor:pointer}
.st-key-wf_step_radio label > div:first-child{display:none}
.st-key-wf_step_radio label:has(input:checked){background:#1B120F;border-color:#1B120F}
.st-key-wf_step_radio label:has(input:checked) p{color:#FBF6EC !important;font-weight:800}
.wfb{display:inline-block;font-size:.66rem;font-weight:800;letter-spacing:.1em;padding:2px 9px;border-radius:99px;margin-right:6px;border:1px solid #1B120F;color:#1B120F}
.wfb.cl{background:#3F7D58;border-color:#3F7D58;color:#fff} .wfb.or{background:#FFFDF8}
.wfi{border-left:4px solid var(--c);background:#FFFDF8;border-radius:0 14px 14px 0;padding:10px 14px;margin:6px 0;border-top:1px solid #E7DCC8;border-right:1px solid #E7DCC8;border-bottom:1px solid #E7DCC8}
.wfi b{font-size:.92rem} .wfi .s{font-size:.8rem;color:#6e6050;margin-top:2px}
</style>"""


def _fig(f, h=300):
    f.update_layout(height=h, margin=dict(t=44, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="")
    return f


def _badge(kind):
    return "<span class='wfb cl'>CLEANED</span>" if kind == "cleaned" else "<span class='wfb or'>ORIGINAL</span>"


def _info(title, text, color=AMBER):
    st.markdown(f"<div class='wfi' style='--c:{color}'><b>{title}</b><div class='s'>{text}</div></div>", unsafe_allow_html=True)


def _kpi(items):
    for col, (lab, val, sub) in zip(st.columns(len(items)), items):
        col.markdown(f"<div class='card' style='padding:12px 14px'><div class='lab'>{lab}</div><div class='stat' style='font-size:1.55rem'>{val}</div><div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def _df(d, **kw):
    st.dataframe(d, hide_index=True, use_container_width=True, **kw)


def render(S, card, go_to, run_integrity):
    wf = S.setdefault("wf", {})
    S.setdefault("wf_n", 0)  # bumped by "Start over" to get fresh upload widgets
    st.markdown(CSS, unsafe_allow_html=True)

    def goto(step):  # safe from callbacks AND from normal code: applied before the selector is built
        S.wf_pending = step

    if S.get("wf_pending"):
        S.wf_step_radio = S.pop("wf_pending")
    elif "wf_step_radio" not in S and wf.get("step") in STEPS:
        S.wf_step_radio = wf["step"]  # widget state is dropped when the page is left; restore the step
    S.setdefault("wf_step_radio", STEPS[0])
    if S.wf_step_radio not in STEPS:
        S.wf_step_radio = STEPS[0]

    st.markdown("<div class='card'><div class='lab'>DATASET BENCHMARK AUDIT · GUIDED WORKFLOW</div>Upload a dataset, understand its problems, optionally clean it, then train and compare simple models "
                "fairly. Every statistic used for preprocessing is learned from the training split only. Your original file is never changed.</div>", unsafe_allow_html=True)
    step = st.radio("Workflow step", STEPS, horizontal=True, key="wf_step_radio", label_visibility="collapsed")
    i = STEPS.index(step)
    wf["step"] = step

    if "raw" in wf:  # always show which data is in play
        cl = wf.get("clean")
        st.markdown(f"<div style='margin:4px 0 2px'>{_badge('original')}<b>{esc(wf['name'])}</b> · {len(wf['raw']):,} rows × {wf['raw'].shape[1]} cols"
                    + (f" &nbsp;→&nbsp; {_badge('cleaned')}{len(cl):,} rows × {cl.shape[1]} cols" if cl is not None else "")
                    + (f" &nbsp;·&nbsp; target: <b>{esc(wf['target'])}</b>" if wf.get("target") else "") + "</div>", unsafe_allow_html=True)
    elif i > 0:
        st.info("Upload a dataset (or pick a sample) first.")
        st.button("Go to Upload →", on_click=goto, args=(STEPS[0],))
        return

    [_upload, _analyze, _clean, _train, _compare, _export][i](S, wf, goto, run_integrity)


# ------------------------------------------------------------------------------------------------ 1 upload
def _set_dataset(wf, df, name, fid):
    wf.clear()
    wf.update(raw=df, name=name, fid=fid)


def _upload(S, wf, goto, _):
    st.subheader("Upload your dataset")
    n = S.wf_n
    up = st.file_uploader("CSV file", type="csv", key=f"wf_up_{n}")
    sample = st.selectbox("…or try a built-in sample (messy on purpose, so every step has something to show)",
                          ["—", "Customer churn (classification)", "House prices (regression)"], key=f"wf_sample_{n}")
    if up is not None:
        fid = ("file", up.name, up.size)
        if wf.get("fid") != fid:
            try:
                df = W.read_csv_safe(up)
                _set_dataset(wf, df, up.name, fid)
            except ValueError as e:
                st.error(str(e))
            except Exception:
                st.error("Something went wrong while reading this file. Please check that it is a valid CSV.")
    elif sample != "—":
        fid = ("sample", sample)
        if wf.get("fid") != fid:
            _set_dataset(wf, W.make_sample("classification" if "churn" in sample else "regression"), sample + " (sample)", fid)
    if "raw" not in wf:
        st.caption("Supported: CSV files with a header row and at least two columns. Large files are fine; training uses at most 50,000 rows.")
        return
    raw = wf["raw"]
    st.success(f"Loaded **{wf['name']}**. This is your ORIGINAL dataset and it will never be modified.")
    _kpi([("ROWS", f"{len(raw):,}", "original"), ("COLUMNS", raw.shape[1], "original"),
          ("NUMERIC", len(W.analyze(raw)["numeric"]), "columns"), ("MISSING CELLS", f"{int(raw.isna().sum().sum()):,}", "see next step")])
    st.markdown("**Preview** (first 10 rows)")
    st.dataframe(raw.head(10), use_container_width=True)
    st.markdown("**Columns and data types**")
    _df(W.column_overview(raw))
    c1, c2, _ = st.columns([1, 1, 2])
    c1.button("Analyze →", on_click=goto, args=(STEPS[1],), type="primary")

    def reset():
        S.wf.clear(); S.wf_n += 1
    c2.button("Start over", on_click=reset)


# ------------------------------------------------------------------------------------------------ helpers
def _target_picker(wf, df, key, required=False):
    cols = list(df.columns)
    opts = cols if required else ["— none —"] + cols
    cur = wf.get("target")
    idx = opts.index(cur) if cur in opts else (0 if not required else len(cols) - 1)
    t = st.selectbox("Target column (what you want to predict)" + ("" if required else " — optional here, needed before training"), opts, index=idx, key=key)
    new = None if t == "— none —" else t
    if new != wf.get("target"):
        wf["target"] = new
        wf["out"] = None  # old results no longer match
    return new


# ------------------------------------------------------------------------------------------------ 2 analyze
def _analyze(S, wf, goto, _):
    st.subheader("Analyze the dataset")
    raw, cl = wf["raw"], wf.get("clean")
    view = "Original"
    if cl is not None:
        view = st.radio("Dataset to analyze", ["Original", "Cleaned"], horizontal=True, key="wf_an_view")
    df = raw if view == "Original" else cl
    st.markdown(f"{_badge('original' if view == 'Original' else 'cleaned')} {len(df):,} rows × {df.shape[1]} columns", unsafe_allow_html=True)
    target = _target_picker(wf, df, "wf_target_an")
    a = W.analyze(df, target)
    cells = df.shape[0] * df.shape[1]
    _kpi([("MISSING CELLS", f"{a['missing_cells']:,}", f"{a['missing_cells'] / max(cells, 1):.1%} of all cells"), ("ROWS WITH MISSING", f"{a['rows_with_missing']:,}", f"{a['rows_with_missing'] / max(len(df), 1):.1%} of rows"),
          ("DUPLICATE ROWS", f"{a['duplicates']:,}", "exact copies"), ("CONSTANT COLUMNS", len(a["constant"]), "one value only"), ("ID-LIKE COLUMNS", len(a["id_like"]), "possible identifiers")])

    st.markdown("### What we found")
    n_found = 0
    if a["missing_cells"]:
        n_found += 1
        top = ", ".join(f"{esc(r.Column)} ({r['Missing %']}%)" for _, r in a["missing"].head(4).iterrows())
        _info("Missing values", f"{a['missing_cells']:,} empty cells across {len(a['missing'])} column(s): {top}. Models cannot use empty cells directly, so they must be filled in (imputed) or the rows/columns removed. "
              "MLProof can fill them safely during training using only the training data.", AMBER)
    if a["duplicates"]:
        n_found += 1
        _info("Duplicate rows", f"{a['duplicates']:,} rows are exact copies of another row. Duplicates can end up in both the training and test sets, which makes results look better than they are.", AMBER)
    if a["constant"]:
        n_found += 1
        _info("Constant columns", f"{esc(', '.join(a['constant']))} {'holds' if len(a['constant']) == 1 else 'hold'} a single value in every row, so {'it cannot' if len(a['constant']) == 1 else 'they cannot'} help a model.", AMBER)
    if a["id_like"]:
        n_found += 1
        _info("Possible ID columns", "; ".join(f"<b>{esc(c)}</b> ({r})" for c, r in a["id_like"].items()) + ". Identifiers do not describe the thing you predict; MLProof leaves them out of training by default.", AMBER)
    if not a["outliers"].empty:
        n_found += 1
        _info("Possible outliers (IQR rule)", ", ".join(f"{esc(r.Column)} ({r.Outliers})" for _, r in a["outliers"].iterrows()) + ". Outliers may be real, so MLProof reports them but never removes them.", TEAL)
    if not n_found:
        _info("No major data-quality issues detected", "No missing values, duplicates, constant, ID-like or outlier columns were found.", "#3F7D58")

    if a["missing_cells"]:
        st.markdown("**Missing values by column**")
        m = a["missing"]
        c1, c2 = st.columns([3, 2])
        f = px.bar(m.sort_values("Missing %"), x="Missing %", y="Column", orientation="h", color_discrete_sequence=[RED]); c1.plotly_chart(_fig(f, 60 + 34 * len(m)), use_container_width=True)
        with c2:
            _df(m)
    st.markdown("**Column types**")
    st.markdown(f"Numeric ({len(a['numeric'])}): {', '.join(a['numeric']) or '—'}  \nCategorical / text ({len(a['categorical'])}): {', '.join(a['categorical']) or '—'}"
                + (f"  \nDate/time ({len(a['datetime'])}): {', '.join(a['datetime'])}" if a["datetime"] else ""))

    if a["numeric"]:
        st.markdown("### Numeric distributions and outliers")
        dist_opts = [c for c in a["numeric"] if c not in a["id_like"]] + [c for c in a["numeric"] if c in a["id_like"]]
        col = st.selectbox("Column", dist_opts, key="wf_dist_col")
        c1, c2 = st.columns(2)
        c1.plotly_chart(_fig(px.histogram(df, x=col, nbins=30, color_discrete_sequence=[RED], title=f"Distribution of {col}"), 280), use_container_width=True)
        c2.plotly_chart(_fig(px.box(df, x=col, color_discrete_sequence=[INK], title=f"Box plot of {col}"), 280), use_container_width=True)
        if not a["outliers"].empty:
            o = a["outliers"].copy()
            for c in ("Lower fence", "Upper fence", "Min", "Max"):
                o[c] = o[c].round(2)
            st.caption("Outliers = values more than 1.5 × IQR beyond the quartiles.")
            _df(o)

    if target:
        st.markdown("### Target column")
        y = df[target].dropna()
        task, why = W.suggest_task(df[target])
        if task == "classification":
            vc = y.astype(str).value_counts()
            if len(vc) <= 30:
                f = px.bar(x=vc.index, y=vc.values, labels=dict(x=target, y="Rows"), color_discrete_sequence=[RED], title=f"Class distribution of {target}")
                st.plotly_chart(_fig(f, 280), use_container_width=True)
                if len(vc) >= 2:
                    ratio = vc.iloc[0] / vc.iloc[-1]
                    if ratio >= 3:
                        _info("Class imbalance", f"'{vc.index[0]}' is {vc.iloc[0] / len(y):.0%} of the rows and '{vc.index[-1]}' only {vc.iloc[-1] / len(y):.0%} (ratio {ratio:.1f}:1). "
                              "Accuracy can look high on imbalanced data even when the rare class is missed, so MLProof will warn you and show recall and F1 too.", RED)
                    else:
                        _info("Class balance looks reasonable", f"Largest to smallest class ratio is {ratio:.1f}:1.", "#3F7D58")
            else:
                st.info(f"The target has {len(vc)} distinct values, which looks more like a continuous number. Consider regression.")
        else:
            st.plotly_chart(_fig(px.histogram(y, nbins=30, color_discrete_sequence=[RED], title=f"Distribution of {target}", labels=dict(value=target)), 280), use_container_width=True)
        st.caption(f"Suggested task: {task}. {why}")
    st.button("Continue to Clean →", on_click=goto, args=(STEPS[2],), type="primary")


# ------------------------------------------------------------------------------------------------ 3 clean
def _clean(S, wf, goto, _):
    st.subheader("Clean the dataset (optional)")
    raw = wf["raw"]
    st.markdown(f"Review what was found, choose what to change and **preview** it. Nothing happens until you press *Apply*. "
                f"Your {_badge('original')} data is kept untouched and a separate {_badge('cleaned')} copy is created.", unsafe_allow_html=True)
    target = _target_picker(wf, raw, "wf_target_cl")
    if not target:
        st.info("Tip: select the target column so cleaning can protect it. The target column is never removed.")
    a = W.analyze(raw, target)
    opts = {}

    st.markdown("#### 1 · Exact duplicate rows")
    if a["duplicates"]:
        opts["dedupe"] = st.checkbox(f"Remove {a['duplicates']:,} exact duplicate row(s) (keep the first copy)", value=True, key="wf_c_dup")
    else:
        st.caption("No exact duplicate rows found.")
    st.markdown("#### 2 · Constant columns")
    if a["constant"]:
        opts["drop_constant"] = st.multiselect("Columns to remove", a["constant"], default=a["constant"], key="wf_c_const")
    else:
        st.caption("No constant columns found.")
    st.markdown("#### 3 · ID-like columns")
    if a["id_like"]:
        st.caption("; ".join(f"{c}: {r}" for c, r in a["id_like"].items()))
        opts["drop_id"] = st.multiselect("Columns to remove from the cleaned data (they are left out of training anyway)", list(a["id_like"]), default=[], key="wf_c_id")
    else:
        st.caption("No ID-like columns found.")
    st.markdown("#### 4 · Missing values")
    if a["missing_cells"]:
        mode = st.radio("Rows with missing values", [
            "Keep the rows (recommended): missing values are filled in during training, using the training split only",
            "Remove only rows where the target is missing",
            f"Remove every row with any missing value ({a['rows_with_missing']:,} rows)"], key="wf_c_rows")
        opts["row_missing"] = "keep" if mode.startswith("Keep") else ("target" if "target is missing" in mode else "any")
        if opts["row_missing"] == "target" and not target:
            st.warning("Select a target column above to use this option.")
            opts["row_missing"] = "keep"
        if opts["row_missing"] == "any":
            st.warning(f"This removes {a['rows_with_missing'] / len(raw):.0%} of your rows. Make sure that is acceptable.")
        if st.checkbox("Also remove columns that are mostly empty", key="wf_c_sparse_on"):
            opts["sparse_threshold"] = float(st.slider("Remove columns with more than this % missing", 10, 95, 50, 5, key="wf_c_sparse"))
        c1, c2 = st.columns(2)
        wf["impute_num"] = c1.selectbox("Fill missing numbers during training with the", ["median", "mean"], key="wf_imp_num")
        wf["impute_cat"] = c2.selectbox("Fill missing categories during training with", ["most_frequent", "constant"], format_func=lambda v: "the most common value" if v == "most_frequent" else "the label 'Missing'", key="wf_imp_cat")
        st.caption("These fill-in values are learned from the training rows only (never from the test rows) when models are trained.")
    else:
        st.caption("No missing values found.")

    cleaned, log = W.apply_cleaning(raw, opts, target)
    st.markdown("#### Preview of the changes")
    b, af = W.cleaning_preview_counts(raw, target), W.cleaning_preview_counts(cleaned, target)
    _df(pd.DataFrame({"": ["Rows", "Columns", "Missing cells", "Duplicate rows", "Constant columns"],
                      "Original": [b["rows"], b["cols"], b["missing_cells"], b["duplicates"], b["constant"]],
                      "After cleaning": [af["rows"], af["cols"], af["missing_cells"], af["duplicates"], af["constant"]]}))
    _df(pd.DataFrame(log))
    with st.expander("Preview the first rows of the cleaned data"):
        st.dataframe(cleaned.head(8), use_container_width=True)

    def apply():
        wf.update(clean=cleaned, clean_log=log, clean_target=target, use="Cleaned", out=None)
        goto(STEPS[3])

    def skip():
        wf.update(clean=None, clean_log=None, use="Original", out=None)
        goto(STEPS[3])

    def discard():
        wf.update(clean=None, clean_log=None, use="Original", out=None)
    c1, c2, c3 = st.columns([1, 1.3, 1.2])
    nothing = log[0]["Action"] == "No changes"
    c1.button("Apply cleaning", on_click=apply, type="primary", disabled=nothing, help="Nothing to change with the current choices." if nothing else None)
    c2.button("Skip: use the original data", on_click=skip)
    if wf.get("clean") is not None:
        c3.button("Discard cleaned dataset", on_click=discard)
        st.success(f"A cleaned dataset exists ({len(wf['clean']):,} rows × {wf['clean'].shape[1]} columns). Applying again replaces it.")
        _df(pd.DataFrame(wf["clean_log"]))


# ------------------------------------------------------------------------------------------------ 4 train
def _train(S, wf, goto, _):
    st.subheader("Choose the task and train models")
    raw, cl = wf["raw"], wf.get("clean")
    use = "Original"
    if cl is not None:
        use = st.radio("Train on", ["Cleaned", "Original"], horizontal=True, index=0 if wf.get("use", "Cleaned") == "Cleaned" else 1, key="wf_use")
    wf["use"] = use
    df = cl if use == "Cleaned" else raw
    st.markdown(f"Training data: {_badge('cleaned' if use == 'Cleaned' else 'original')} {len(df):,} rows × {df.shape[1]} columns", unsafe_allow_html=True)
    if cl is not None and wf.get("clean_target") and wf.get("target") and wf["clean_target"] != wf["target"]:
        st.warning(f"The cleaned dataset was created with '{wf['clean_target']}' as the protected target, but the target is now '{wf['target']}'. Re-apply cleaning if you want it protected.")

    target = _target_picker(wf, df, "wf_target_tr", required=True)
    if target not in df.columns:
        st.error("The selected target is not in this dataset.")
        return
    sug, why = W.suggest_task(df[target])
    c1, c2 = st.columns([2, 3])
    c1.markdown(f"<div class='card'><div class='lab'>SUGGESTED TASK</div><div class='stat' style='font-size:1.4rem'>{sug.title()}</div><div class='sub'>{why}</div></div>", unsafe_allow_html=True)
    choice = c2.radio("Task (you can correct the suggestion)", ["Use the suggestion", "classification", "regression"], key=f"wf_task_{target}")
    task = sug if choice.startswith("Use") else choice
    if task != wf.get("task"):
        wf["task"] = task; wf["out"] = None
    err, warn, _i = W.validate_target(df, target, task)
    for e in err:
        st.error(e)
    for w in warn:
        st.warning(w)
    if err:
        return

    usable, excluded, leaks = W.feature_report(df, target)
    default = [c for c in usable if c not in leaks]
    feats = st.multiselect(f"Features used to predict '{target}'", usable, default=default, key=f"wf_feat|{target}|{use}")
    if excluded:
        with st.expander(f"{len(excluded)} column(s) left out automatically"):
            _df(pd.DataFrame({"Column": list(excluded), "Reason": list(excluded.values())}))
    if leaks:
        st.warning("Possible target leakage (heuristic, not proof) — left out by default: " + "; ".join(f"**{c}** ({r})" for c, r in leaks.items()) +
                   ". A feature that is derived from the answer makes results look unrealistically good. Add it back only if you are sure it is available at prediction time.")

    avail = W.model_availability()
    models = [m for m in (W.CLASS_MODELS if task == "classification" else W.REG_MODELS) if not avail.get(m)]
    sel = st.multiselect("Models to compare", models, default=[m for m in models if m not in W.DEFAULT_OFF], key=f"wf_models_{task}",
                         help="TabNet is an optional deep-learning model: it is slower and is off by default.")
    missing = {m: r for m, r in avail.items() if r}
    if missing:
        st.caption("Not available in this environment: " + "; ".join(f"**{m}** ({r})" for m, r in missing.items()) + ". The other models work without them.")
    positive = None
    if task == "classification" and df[target].nunique() == 2:
        vals = [str(v) for v in df[target].dropna().unique()]
        minority = df[target].dropna().astype(str).value_counts().index[-1]
        positive = st.selectbox("Positive class for precision, recall and F1", sorted(vals), index=sorted(vals).index(minority), key=f"wf_pos_{target}",
                                help="Usually the rarer or more important outcome. Multi-class targets use a macro average instead.")
    with st.expander("Model settings"):
        d = W.default_params()
        c1, c2, c3 = st.columns(3)
        ts = c1.slider("Test set size", 10, 40, 25, 5, format="%d%%", key="wf_ts") / 100
        seed = int(c2.number_input("Random seed", 0, 99999, d["seed"], key="wf_seed"))
        td = int(c3.slider("Decision tree max depth", 2, 20, d["tree_depth"], key="wf_td"))
        c4, c5, _c = st.columns(3)
        rt = int(c4.slider("Random forest trees", 20, 300, d["rf_trees"], 10, key="wf_rt"))
        rd = int(c5.number_input("Random forest max depth (0 = unlimited)", 0, 40, 0, key="wf_rd"))
        c6, c7, c8 = st.columns(3)
        xt = int(c6.slider("XGBoost trees", 50, 500, d["xgb_trees"], 50, key="wf_xt"))
        ci = int(c7.slider("CatBoost iterations", 50, 500, d["cat_iters"], 50, key="wf_ci"))
        te = int(c8.slider("TabNet max epochs", 5, 100, d["tabnet_epochs"], 5, key="wf_te"))
        st.caption("All models share one train/test split. Missing-value filling, scaling and encoding are fitted on the training rows only.")
    params = dict(test_size=ts, seed=seed, tree_depth=td, rf_trees=rt, rf_depth=rd or None, xgb_trees=xt, cat_iters=ci, tabnet_epochs=te, impute_num=wf.get("impute_num", "median"), impute_cat=wf.get("impute_cat", "most_frequent"))

    if not feats:
        st.error("Select at least one feature.")
        return
    if not sel:
        st.error("Select at least one model.")
        return
    if st.button("TRAIN & COMPARE →", type="primary"):
        bar = st.progress(0.0, text="Preparing data…")
        try:
            with st.spinner("Training models…"):
                out = W.train_and_compare(df, target, task, feats, sel, params, positive, progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
            wf.update(out=out, out_meta=dict(use=use, rows=len(df), cols=df.shape[1]), params=params)
            bar.empty()
            goto(STEPS[4]); st.rerun()
        except ValueError as e:
            bar.empty(); st.error(str(e))
        except Exception:
            bar.empty(); st.error("Training failed for this dataset. Check the target, task type and selected features.")
    elif wf.get("out"):
        st.caption("Results from your last training run are available in the Compare step. Press Train again to refresh them.")


# ------------------------------------------------------------------------------------------------ 5 compare
def _compare(S, wf, goto, _):
    st.subheader("Compare models")
    out = wf.get("out")
    if not out:
        st.info("Train models first.")
        st.button("Go to Train →", on_click=goto, args=(STEPS[3],))
        return
    meta = wf.get("out_meta", {})
    task = out["task"]
    st.markdown(f"{_badge('cleaned' if meta.get('use') == 'Cleaned' else 'original')} target <b>{esc(out['target'])}</b> · {task} · {len(out['features'])} features · "
                f"{out['n_train']:,} training rows / {out['n_test']:,} test rows · seed {out['params']['seed']}", unsafe_allow_html=True)
    msgs, warns = W.explain(out)
    for w in warns:
        _info("Warning: accuracy may be misleading" if "misleading" in w else "Warning", w, RED)
    for n in out["notes"]:
        st.caption("ℹ️ " + n)
    for k, v in out["failed"].items():
        st.warning(f"{k} could not be trained: {v}")

    tbl = W.results_table(out)
    best, crit = W.best_model(out)
    st.markdown(f"**Results on the held-out test set**  ·  best non-baseline model by {crit}: **{best}**")
    _df(tbl)
    if task == "classification":
        st.caption(f"Precision, recall and F1 use the {out['averaging']}. Balanced accuracy averages recall over classes.")

    if task == "classification":
        long = tbl.melt("Model", ["Accuracy", "Balanced accuracy", "Precision", "Recall", "F1"], var_name="Metric", value_name="Score")
        f = px.bar(long, x="Metric", y="Score", color="Model", barmode="group", color_discrete_map=MODEL_COL, color_discrete_sequence=PAL, title="Classification metrics by model (higher is better)")
        f.update_yaxes(range=[0, 1.05]); st.plotly_chart(_fig(f, 340), use_container_width=True)
    else:
        c1, c2 = st.columns(2)
        c1.plotly_chart(_fig(px.bar(tbl, x="Model", y=["MAE", "RMSE"], barmode="group", color_discrete_sequence=[RED, INK], title="Error (lower is better)", labels=dict(value="Error")), 320), use_container_width=True)
        c2.plotly_chart(_fig(px.bar(tbl, x="Model", y="R2", color_discrete_sequence=[AMBER], title="R² (higher is better; 0 = predicting the average)"), 320), use_container_width=True)

    st.markdown("### Model detail")
    name = st.selectbox("Model", list(out["results"]), index=list(out["results"]).index(best), key="wf_detail")
    r = out["results"][name]
    if task == "classification":
        c1, c2 = st.columns(2)
        c1.plotly_chart(_fig(px.imshow(r["confusion"], x=out["classes"], y=out["classes"], text_auto=True, color_continuous_scale="Reds", title="Confusion matrix (rows = actual, columns = predicted)",
                                       labels=dict(x="Predicted", y="Actual")), 340), use_container_width=True)
        with c2:
            st.markdown("**Per-class results**")
            _df(pd.DataFrame(r["per_class"]).round(3))
    else:
        c1, c2 = st.columns(2)
        yt, yp = r["y_true"], r["y_pred"]
        f = px.scatter(x=yt, y=yp, labels=dict(x="Actual", y="Predicted"), color_discrete_sequence=[RED], title="Predicted vs actual (test set)")
        lo, hi = min(min(yt), min(yp)), max(max(yt), max(yp)); f.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=INK, dash="dash"), name="perfect"))
        c1.plotly_chart(_fig(f, 340), use_container_width=True)
        c2.plotly_chart(_fig(px.histogram(x=[p - t for t, p in zip(yt, yp)], nbins=30, color_discrete_sequence=[INK], title="Residuals (predicted − actual)", labels=dict(x="Residual")), 340), use_container_width=True)

    st.markdown("### What this means")
    for m in msgs:
        st.markdown("- " + m)
    st.button("Continue to Export →", on_click=goto, args=(STEPS[5],), type="primary")


# ------------------------------------------------------------------------------------------------ 6 export
def _export(S, wf, goto, run_integrity):
    st.subheader("Export")
    out, cl = wf.get("out"), wf.get("clean")
    base = "".join(ch if ch.isalnum() else "_" for ch in wf["name"].rsplit(".", 1)[0]).strip("_") or "dataset"
    analysis = W.analyze(wf["raw"], wf.get("target"))
    use_label = "Cleaned dataset" if (out and wf.get("out_meta", {}).get("use") == "Cleaned") else "Original dataset"
    summary = W.summary_dict(wf["name"], analysis, wf.get("clean_log"), list(cl.shape) if cl is not None else None, use_label, out)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='card'><div class='lab'>CLEANED DATA</div>{_badge('cleaned')}<div class='sub' style='margin-top:6px'>The cleaned copy as a CSV. Your original file is unchanged.</div></div>", unsafe_allow_html=True)
        if cl is not None:
            st.download_button("Download cleaned CSV", cl.to_csv(index=False).encode("utf-8"), f"{base}_cleaned.csv", "text/csv", key="wf_dl_clean")
        else:
            st.caption("No cleaned dataset yet. Apply cleaning in step 3 first.")
    with c2:
        st.markdown("<div class='card'><div class='lab'>MODEL COMPARISON</div><div class='sub'>One row per model with every metric, as a CSV.</div></div>", unsafe_allow_html=True)
        if out:
            st.download_button("Download comparison CSV", W.results_table(out).to_csv(index=False).encode("utf-8"), f"{base}_model_comparison.csv", "text/csv", key="wf_dl_cmp")
        else:
            st.caption("Train models in step 4 first.")
    with c3:
        st.markdown("<div class='card'><div class='lab'>SUMMARY REPORT</div><div class='sub'>Dataset findings, cleaning actions, model settings and evaluation metrics.</div></div>", unsafe_allow_html=True)
        if out:
            st.download_button("Download report (PDF)", R.to_pdf(summary), f"{base}_model1_report.pdf", "application/pdf", key="wf_dl_pdf")
            st.download_button("Download report (JSON)", R.to_json(summary), f"{base}_model1_report.json", "application/json", key="wf_dl_json")
        else:
            st.caption("Train models in step 4 first.")
    if out:
        st.markdown("### Optional: run the MLProof integrity audit")
        st.caption("Uses the existing integrity audit (leakage and contamination heuristics, cross-validation stability, claim challenge, dataset discovery) on the dataset you trained on.")
        df = cl if wf.get("out_meta", {}).get("use") == "Cleaned" else wf["raw"]
        if st.button("Run integrity audit on this dataset →"):
            run_integrity(df, wf["name"] + (" (cleaned)" if df is cl else ""), out["target"], out["task"])
