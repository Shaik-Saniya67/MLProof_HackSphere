import pandas as pd, plotly.express as px, streamlit as st
from core.actual_audit import audit_predictions

def render(S, card, SEVC, RED):
    st.markdown("<div class='card'><div class='lab'>ACTUAL MODEL AUDIT</div>Upload the <b>actual predictions</b> of an already-trained model next to the ground-truth test labels. MLProof recomputes the metrics independently - <b>no model is trained</b>. "
                "This audits the supplied evaluation results, not every aspect of the model's development; it cannot prove the test set was unseen or that no leakage occurred.</div>", unsafe_allow_html=True)
    up = st.file_uploader("Predictions CSV", type="csv", key="act_up")
    if up is not None:
        try:
            df = pd.read_csv(up)
            if df.empty or df.shape[1] < 2: raise ValueError
        except Exception: st.error("Could not read that file as a valid CSV with at least two columns."); df = None
        if df is not None:
            cols = list(df.columns); c = st.columns(2)
            pc = c[0].selectbox("Prediction column", cols, index=0, key="act_p"); tc = c[1].selectbox("Ground-truth column", cols, index=min(1, len(cols) - 1), key="act_t")
            task = st.radio("Task type", ["classification", "regression"], horizontal=True, key="act_task")
            sc = None
            if task == "classification":
                sc = st.selectbox("Probability / score column (optional, binary only)", ["—"] + cols, key="act_s"); sc = None if sc == "—" else sc
            claim = st.text_input("Performance claim (optional)", placeholder="My model achieves 96% accuracy.", key="act_claim")
            if st.button("RUN ACTUAL MODEL AUDIT →"):
                try:
                    S.actual = audit_predictions(df, pc, tc, task, claim, sc); S.actual["name"] = up.name
                    S.history.append(dict(name=up.name + " (actual model)", status=S.actual["status"], task=task, issues=len([f for f in S.actual["findings"] if f["severity"] != "INFO"]), t=__import__("time").time()))
                except ValueError as e: st.error(str(e)); S.actual = None
                except Exception: st.error("The audit could not be completed. Check the selected columns and task type."); S.actual = None
    a = S.get("actual")
    if not a: return
    m = a["metrics"]; st.markdown(f"<div class='lab'>ACTUAL MODEL AUDIT - {a['name'].upper()}</div><div class='status'>{a['status']}</div>", unsafe_allow_html=True)
    st.caption(f"{a['n_used']:,} of {a['n_rows']:,} rows used")
    keys = [k for k in (["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"] if a["task"] == "classification" else ["mae", "mse", "rmse", "r2"]) if k in m]
    for col, k in zip(st.columns(len(keys)), keys): 
        with col: card(k.upper().replace("_", " "), f"{m[k]*100:.1f}%" if a["task"] == "classification" else f"{m[k]:.3f}")
    if a["task"] == "classification":
        st.caption(f"Precision/recall/F1: {a['averaging']}. {a['score_note'] or ''}")
    cl = a["claim"]
    if cl["text"].strip():
        st.markdown("## PERFORMANCE CLAIM CHALLENGE"); r = cl.get("result")
        body = f"<div class='lab'>CLAIM</div><div style='font-size:1.3rem;font-weight:800'>“{cl['text']}”</div>"
        if r: body += f"<br>Recomputed from your predictions: <b>{r['computed']*100 if a['task']=='classification' else r['computed']:.4g}{'%' if a['task']=='classification' else ''}</b> vs claimed <b>{r['claimed']*100 if a['task']=='classification' else r['claimed']:.4g}{'%' if a['task']=='classification' else ''}</b>."
        else: body += f"<br>{cl.get('note') or ''}"
        st.markdown(f"<div class='card'>{body}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card'><div class='status' style='font-size:1.5rem'>{a['status']}</div>{a['conclusion']}</div>", unsafe_allow_html=True)
    pal = [RED, "#1B120F"]; g = st.columns(2)
    if a["task"] == "classification":
        g[0].plotly_chart(px.imshow(a["confusion"], x=a["classes"], y=a["classes"], text_auto=True, title="Confusion matrix (rows = true)", color_continuous_scale="Reds"), use_container_width=True)
        pcdf = pd.DataFrame(a["per_class"]).T.reset_index().melt("index", ["precision", "recall", "f1"])
        g[1].plotly_chart(px.bar(pcdf, x="index", y="value", color="variable", barmode="group", title="Per-class metrics", color_discrete_sequence=[RED, "#1B120F", "#C98A1B"]), use_container_width=True)
    else:
        g[0].plotly_chart(px.scatter(x=a["y_true"], y=a["y_pred"], labels=dict(x="True", y="Predicted"), title="Predicted vs true", color_discrete_sequence=pal), use_container_width=True)
        g[1].plotly_chart(px.histogram(x=a["residuals"], title="Residuals", color_discrete_sequence=pal), use_container_width=True)
    st.subheader("Key findings")
    for f in a["findings"]:
        st.markdown(f"<div class='card'><span class='badge' style='background:{SEVC[f['severity']]}'>{f['severity']}</span> <b>{f['title']}</b><br><b>Evidence:</b> {f['evidence']}<br><b>Why it matters:</b> {f['why']}<br><b>Recommendation:</b> {f['recommendation']}</div>", unsafe_allow_html=True)
    st.caption("Export from the Reports page.")
