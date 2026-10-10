import io, time, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from core.audit import run_audit, detect_task, profile
from core.discovery import recommend, needs_from, load
from core.demo import demo
from reports.report_generator import to_pdf, to_json, actual_to_pdf
from core import actual_ui, workflow_ui

st.set_page_config("MLProof", "✔", layout="wide")
RED = "#D8232A"
st.markdown(f"""<style>
.stApp{{background:#FBF6EC;color:#1B120F}} h1,h2,h3{{font-weight:900;letter-spacing:-.02em}}
.hero{{font-size:4rem;font-weight:900;line-height:1;letter-spacing:-.03em}} .red{{color:{RED}}}
.card{{background:#FFFDF8;border:1px solid #E7DCC8;border-radius:18px;padding:18px 20px;margin:8px 0;box-shadow:0 6px 20px rgba(60,30,10,.06)}}
.lab{{font-size:.7rem;letter-spacing:.14em;color:#8a7b6a;font-weight:700}} .stat{{font-size:2rem;font-weight:800}}
.badge{{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.7rem;font-weight:800;color:#fff}}
.status{{font-size:2.6rem;font-weight:900;color:{RED}}}
.stButton>button{{border-radius:10px;border:1px solid #1B120F;font-weight:700}}

.kpi{{background:#FFFDF8;border:1px solid #E7DCC8;border-left:5px solid var(--c);border-radius:16px;padding:16px 18px;transition:.2s}}
.kpi:hover{{transform:translateY(-3px);box-shadow:0 10px 24px rgba(60,30,10,.10)}}
.kpi .n{{font-size:2.3rem;font-weight:900;line-height:1.1}} .kpi .i{{float:right;font-size:1.3rem}} .sub{{font-size:.78rem;color:#8a7b6a}}
.ladder{{display:flex;gap:6px;margin:10px 0 6px}} .seg{{flex:1;height:10px;border-radius:6px;background:#E7DCC8}}
.feed{{display:flex;justify-content:space-between;align-items:center;padding:12px 4px;border-bottom:1px solid #E7DCC8}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px}}
.band{{background:#1B120F;color:#FBF6EC;border-radius:20px;padding:26px 30px;margin:14px 0}} .band .lab{{color:#d9b9a0}}
.rule{{border-top:1px solid #E7DCC8;margin:22px 0 10px}}
[data-testid=stSidebar]{{background:#F4EDDD;border-right:1px solid #E7DCC8}}
.mlp-foot{{margin-top:14px;padding-top:10px;border-top:1px solid #D9CDB5}}
[data-testid=stSidebar] [data-testid=stVerticalBlock]{{min-height:calc(100vh - 4.2rem)}}
[data-testid=stSidebar] [data-testid=stElementContainer]:has(.mlp-foot){{margin-top:auto}}
.mlp-foot .fh{{font-size:.78rem;font-weight:800;letter-spacing:.14em;color:#1B120F}} .mlp-foot .fs{{font-size:.68rem;font-weight:400;color:#8a7b6a;margin-top:2px}}
[data-testid=stSidebarHeader]{{height:2rem;min-height:2rem;padding:.4rem 1rem 0}}
[data-testid=stSidebarUserContent]{{padding:0 1.25rem 1rem}}
[data-testid=stSidebar] [data-testid=stVerticalBlock]{{gap:.1rem}}
[data-testid=stSidebar] .stButton>button{{min-height:2rem;padding:2px 10px}}
[data-testid=stSidebar] .stButton>button{{background:transparent;border:none;border-left:3px solid transparent;border-radius:0;justify-content:flex-start;text-align:left;font-weight:500;padding:4px 10px;color:#1B120F;box-shadow:none}}
[data-testid=stSidebar] .stButton>button:hover{{background:#EBE1CC;color:#1B120F}}
[data-testid=stSidebar] .stButton>button[kind=primary],[data-testid=stSidebar] [data-testid=stBaseButton-primary]{{background:#FFFDF8;border-left:3px solid {RED};color:{RED};font-weight:800}}</style>""", unsafe_allow_html=True)
SEVC = dict(INFO="#8a7b6a", LOW="#4a7c59", MEDIUM="#c98a1b", HIGH="#D8232A", CRITICAL="#7a0c10")
S = st.session_state
for k, v in dict(df=None, name=None, audit=None, recs=None, history=[], audits={}, actual=None).items(): S.setdefault(k, v)
card = lambda lab, val: st.markdown(f"<div class='card'><div class='lab'>{lab}</div><div class='stat'>{val}</div></div>", unsafe_allow_html=True)
def go_to(p): S.pending_nav = p
S.setdefault("nav", "Home")
if S.get("pending_nav"): S.nav = S.pop("pending_nav")  # must run BEFORE any nav button is rendered
page = S.nav
GROUPS = [("MAIN", ["Home", "Dashboard", "New Audit"]), ("ANALYZE", ["Dataset", "Evaluation", "Findings"]), ("DISCOVER", ["Dataset Discovery"]), ("OUTPUT", ["Reports"])]
with st.sidebar:
    st.markdown("<div style='font-size:2.3rem;font-weight:900;line-height:.92;letter-spacing:-.03em;margin-top:6px'>ML<br><span class='red'>PROOF</span></div><div class='sub' style='margin:8px 0 4px'>Train, Compare, and Verify Machine Learning Models.</div><div class='rule' style='margin:12px 0 4px'></div>", unsafe_allow_html=True)
    for g, items in GROUPS:
        st.markdown(f"<div class='lab' style='margin:10px 0 0;font-size:.62rem'>{g}</div>", unsafe_allow_html=True)
        for p in items:
            st.button(p, key=f"nav_{p}", on_click=go_to, args=(p,), type="primary" if p == page else "secondary", use_container_width=True)
    st.markdown("<div class='mlp-foot'><div class='fh'>HACKSPHERE 2026</div><div class='fs'>SITAMS, Chittoor</div></div>", unsafe_allow_html=True)

def load_df(df, name):
    S.df, S.name, S.audit, S.recs = df, name, None, None

def run(df, name, target, task):
    with st.spinner("Auditing evidence..."):
        try:
            a = run_audit(df, target, task); a["name"] = name; S.audit = a; S.audits[name] = a; S.recs = recommend(a)
            S.history.append(dict(name=name, status=a["status"], task=a["task"], issues=len([f for f in a["findings"] if f["severity"] != "INFO"]), t=time.time())); return True
        except ValueError as e: st.error(str(e))
        except Exception: st.error("The audit could not be completed for this dataset. Check the target column and data types.")
    return False

def need_audit():
    if S.audit is None: st.info("Run an audit first (New Audit)."); st.button("Go to New Audit", on_click=go_to, args=("New Audit",)); return True

def try_demo_button():
    if st.button("Try Demo Dataset →"):
        load_df(demo("imbalanced"), "Imbalanced demo")
        if run(S.df, S.name, "label", "classification"): go_to("Evaluation"); st.rerun()

if page == "Home":
    l, r = st.columns([3, 2])
    with l:
        st.markdown("<div class='lab'>ML EVALUATION INTEGRITY AUDITOR</div><div class='hero'>ML<span class='red'>Proof</span><br>DON'T TRUST<br>THE ACCURACY.<br><span class='red'>AUDIT THE EVIDENCE.</span></div>", unsafe_allow_html=True)
        st.write("MLProof checks whether an ML model's reported performance is actually supported by reliable evaluation.")
        c1, c2, c3, _ = st.columns([1, 1, 1, 0.6]); c1.button("Start an Audit →", on_click=go_to, args=("New Audit",))
        with c2: try_demo_button()
        c3.button("Discover Datasets →", on_click=go_to, args=("Dataset Discovery",))
    with r:
        st.markdown("<div class='card'><div class='lab'>SAMPLE FINDING</div><div class='stat red'>SIGNIFICANT CONCERNS</div>Accuracy may overstate quality when the minority class is only 6% of the data. Run the demo to see real numbers.</div>", unsafe_allow_html=True)
    st.markdown("### 01 — The problem\n**A high accuracy score can still hide a bad model.** With 94% Class A and 6% Class B, predicting Class A every time scores 94%.")
    st.markdown("### 02 — What MLProof checks")
    cs = st.columns(3)
    for i, t in enumerate(["Data Quality", "Class Balance", "Leakage", "Contamination", "Validation Stability", "Baseline Comparison"]): cs[i % 3].markdown(f"<div class='card'><span class='red'>0{i+1}</span> {t}</div>", unsafe_allow_html=True)
    st.markdown("### 03 — Performance Claim Challenge\n*“Our model achieves 96% accuracy.”* vs **Does the evidence actually support this claim?**")
    st.markdown("### 04 — Dataset discovery\nYour dataset has problems → MLProof identifies what is missing → discover alternatives → compare and re-audit.")
    st.markdown("### 05 — How it works\n**AUDIT → DIAGNOSE → DISCOVER → RECOMMEND → RE-EVALUATE**")
    st.markdown("### 06 — Before you trust a model, <span class='red'>test the evidence behind it.</span>", unsafe_allow_html=True)

elif page == "Dashboard":
    LV = [("RELATIVELY SOUND", "#3F7D58", "🟢"), ("REVIEW RECOMMENDED", "#C98A1B", "🟡"), ("SIGNIFICANT CONCERNS", "#D9631E", "🟠"), ("HIGH-RISK EVALUATION", RED, "🔴")]
    ago = lambda t: "just now" if time.time() - t < 60 else (f"{int((time.time()-t)//60)} min ago" if time.time() - t < 3600 else f"{int((time.time()-t)//3600)} h ago")
    A = list(S.audits.values()); cur = S.audit
    l, r = st.columns([3, 2])
    with l:
        st.markdown("<div class='lab'>ML EVALUATION INTEGRITY</div><div style='font-size:3rem;font-weight:900;line-height:1.02;letter-spacing:-.03em'>ML<span class='red'>Proof</span></div><div style='font-size:1.35rem;margin-top:8px'>Measure the model.<br>Question the evidence.<br><b>Improve the evaluation.</b></div>", unsafe_allow_html=True)
    nissues = sum(len([f for f in a["findings"] if f["severity"] != "INFO"]) for a in A)
    ncrit = sum(f["severity"] in ("HIGH", "CRITICAL") for a in A for f in a["findings"])
    kp = [("AUDITS COMPLETED", len(S.history), "📋", "#1B120F", "runs this session"), ("ISSUES DETECTED", nissues, "⚠️", "#C98A1B", "excluding informational"),
          ("CRITICAL FINDINGS", ncrit, "🚩", RED, "HIGH or CRITICAL severity"), ("DATASETS DISCOVERED", len(S.recs or []), "🧭", "#1F6F6B", "candidates matched to your audit")]
    for c, (lab, n, ic, col, sub) in zip(st.columns(4), kp):
        c.markdown(f"<div class='kpi' style='--c:{col}'><span class='i'>{ic}</span><div class='lab'>{lab}</div><div class='n'>{n}</div><div class='sub'>{sub}</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    L, R = st.columns([2, 3])
    with L:
        st.markdown("<div class='lab'>AUDIT HEALTH</div>", unsafe_allow_html=True)
        if cur:
            i = [x[0] for x in LV].index(cur["status"]); _, col, em = LV[i]
            segs = "".join(f"<div class='seg' style='background:{LV[j][1] if j <= i else ''}'></div>" for j in range(4))
            st.markdown(f"<div class='card'><div class='lab'>{cur['name'].upper()}</div><div style='font-size:1.7rem;font-weight:900;color:{col}'>{em} {cur['status']}</div><div class='ladder'>{segs}</div><div class='sub'>Sound → Review → Significant → High-risk</div><p style='margin-top:10px'>{cur['claim_conclusion']}</p></div>", unsafe_allow_html=True)
            st.button("Open evaluation →", on_click=go_to, args=("Evaluation",))
        else:
            st.markdown("<div class='card'><div style='font-size:1.4rem;font-weight:800'>No audit yet</div><div class='sub'>Status appears here once an audit has run.</div><div class='ladder'>" + "<div class='seg'></div>" * 4 + "</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='lab' style='margin-top:18px'>RECENT AUDITS</div>", unsafe_allow_html=True)
        if S.history:
            rows = ""
            for h in reversed(S.history[-6:]):
                col = next(c for n, c, _ in LV if n == h["status"])
                rows += f"<div class='feed'><div><b>{h['name']}</b><br><span class='sub'>{h['task'].title()} · {h['issues']} issues · {ago(h['t'])}</span></div><div style='color:{col};font-weight:800;font-size:.75rem;text-align:right'><span class='dot' style='background:{col}'></span>{h['status'].title()}</div></div>"
            st.markdown(f"<div class='card' style='padding:6px 18px'>{rows}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card' style='text-align:center;padding:30px'><div style='font-size:1.6rem;font-weight:900'>No audits yet.</div><div class='sub' style='margin:6px 0 4px'>Upload your first dataset and challenge its performance.</div></div>", unsafe_allow_html=True)
            st.button("START YOUR FIRST AUDIT →", on_click=go_to, args=("New Audit",))
    with R:
        st.markdown("<div class='lab'>EVIDENCE AT A GLANCE</div>", unsafe_allow_html=True)
        if cur:
            m = cur["models"][cur["main"]]; mk = "accuracy" if cur["task"] == "classification" else "r2"
            sc = {k: sum(f["severity"] == k for f in cur["findings"]) for k in ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]}
            f1 = px.bar(x=list(sc), y=list(sc.values()), title="Findings by severity", color=list(sc), color_discrete_map=SEVC)
            f2 = px.bar(x=list(cur["models"]), y=[v[mk] for v in cur["models"].values()], title=f"Model vs baseline ({mk})", color_discrete_sequence=[RED])
            for f in (f1, f2): f.update_layout(showlegend=False, height=250, margin=dict(t=40, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title=None, yaxis_title=None)
            c1, c2 = st.columns(2); c1.plotly_chart(f1, use_container_width=True); c2.plotly_chart(f2, use_container_width=True)
            if "class_dist" in cur:
                f3 = px.bar(x=list(cur["class_dist"].values()), y=list(cur["class_dist"]), orientation="h", title=f"Class balance — {cur['imbalance_ratio']:.1f}:1", color_discrete_sequence=["#1B120F"])
                f3.update_layout(height=200, margin=dict(t=40, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title=None, yaxis_title=None); st.plotly_chart(f3, use_container_width=True)
        else:
            st.markdown("<div class='card sub'>Charts (severity mix, model vs baseline, class balance) appear after your first audit.</div>", unsafe_allow_html=True)
    st.markdown("<div class='band'><div class='lab'>DISCOVER BETTER DATA</div><div style='font-size:1.8rem;font-weight:900;line-height:1.15'>Found weaknesses in your dataset?<br>Find alternatives that address them.</div></div>", unsafe_allow_html=True)
    st.button("EXPLORE DATASETS →", on_click=go_to, args=("Dataset Discovery",))

elif page == "New Audit":
    st.title("New Audit")
    mode = st.radio("Audit mode", ["Dataset Benchmark Audit", "Actual Model Audit"], horizontal=True, key="audit_mode")
    if mode == "Actual Model Audit": actual_ui.render(S, card, SEVC, RED); st.stop()
    view = st.radio("View", ["Guided workflow", "Quick integrity audit"], horizontal=True, key="m1_view")
    if view == "Guided workflow":
        def _m1_integrity(df, name, target, task):
            load_df(df, name)
            if run(df, name, target, task): go_to("Evaluation"); st.rerun()
        workflow_ui.render(S, card, go_to, _m1_integrity); st.stop()
    up = st.file_uploader("Upload CSV", type="csv")
    demos = {"Balanced": "balanced", "Imbalanced (94:6)": "imbalanced", "Overfitting": "overfit", "Contamination": "contaminated"}
    d = st.selectbox("…or use a demo dataset", ["—"] + list(demos))
    if up is not None:
        try:
            df = pd.read_csv(up)
            if df.empty: raise ValueError
            if S.name != up.name: load_df(df, up.name)
        except Exception: st.error("Could not read that file as a valid, non-empty CSV.")
    elif d != "—" and S.name != d: load_df(demo(demos[d]), d)
    if S.df is not None:
        df = S.df; guess = "label" if "label" in df.columns else df.columns[-1]
        t = st.selectbox("Target column", df.columns, index=list(df.columns).index(guess))
        task = st.radio("Problem type", ["auto", "classification", "regression"], horizontal=True)
        det = detect_task(df[t]); p = task if task != "auto" else det
        c = st.columns(4)
        for col, (l, v) in zip(c, [("DATASET", S.name), ("ROWS", f"{len(df):,}"), ("COLUMNS", df.shape[1]), ("PROBLEM TYPE", p)]): col.markdown(f"<div class='card'><div class='lab'>{l}</div><b>{v}</b></div>", unsafe_allow_html=True)
        if p == "classification" and df[t].nunique() > 50: st.warning("Target has many distinct values; consider regression.")
        if st.button("RUN INTEGRITY AUDIT →"):
            if run(df, S.name, t, p): go_to("Evaluation"); st.rerun()

elif page == "Dataset":
    st.title("Dataset")
    if S.df is None: st.info("Load a dataset in New Audit.")
    else:
        P = profile(S.df, S.audit["target"] if S.audit else S.df.columns[-1]); c = st.columns(4)
        for col, (l, v) in zip(c, [("ROWS", f"{P['rows']:,}"), ("COLUMNS", P["cols"]), ("MISSING", f"{P['missing_pct']:.1f}%"), ("DUPLICATES", P["duplicates"])]):
            with col: card(l, v)
        st.write(f"**Numeric:** {P['numeric']}  \n**Categorical:** {P['categorical']}  \n**ID-like:** {P['id_cols']}  \n**Constant:** {P['constant']}  \n**Near-constant:** {P['near_constant']}")
        st.dataframe(S.df.head(50))

elif page == "Evaluation":
    st.title("Evaluation")
    if not need_audit():
        a = S.audit; m = a["models"][a["main"]]; base = a["models"][list(a["models"])[0]]
        st.markdown(f"<div class='lab'>EVALUATION INTEGRITY — {a['name']}</div><div class='status'>{a['status']}</div>", unsafe_allow_html=True)
        if a.get("sampled"): st.caption("Large dataset: audited on a 50,000-row sample.")
        keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"] if a["task"] == "classification" else ["mae", "mse", "rmse", "r2"]
        cs = st.columns(len(keys))
        for c, k in zip(cs, [k for k in keys]):
            with c: card(k.upper().replace("_", " "), f"{m[k]*100:.1f}%" if a["task"] == "classification" and k in m else (f"{m[k]:.3f}" if k in m else "n/a"))
        st.markdown("## PERFORMANCE CLAIM CHALLENGE")
        mk = "accuracy" if a["task"] == "classification" else "r2"
        st.markdown(f"<div class='card'><div class='lab'>CLAIM</div><div style='font-size:1.4rem;font-weight:800'>“Our model achieves {m[mk]*100:.0f}% {'accuracy' if mk=='accuracy' else 'R²'}.”</div><br><b>MLProof asks:</b> Does the available evidence actually support this claim?</div>", unsafe_allow_html=True)
        ev = [(mk.upper(), f"{m[mk]*100:.1f}%"), ("Baseline", f"{base[mk]*100:.1f}%"), ("Improvement", f"{a['improvement']*100:+.1f} pts"), ("CV std", f"{a['cv']['std']:.3f}"), ("Train–test gap", f"{a['gap']:.3f}")]
        if a["task"] == "classification": ev[1:1] = [("F1", f"{m['f1']*100:.1f}%"), ("Minority recall", f"{m['minority_recall']*100:.1f}%"), ("Imbalance", f"{a['imbalance_ratio']:.1f}:1")]
        for col, (l, v) in zip(st.columns(len(ev)), ev): col.markdown(f"<div class='card'><div class='lab'>{l}</div><b>{v}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><div class='status' style='font-size:1.6rem'>{a['status']}</div>{a['claim_conclusion']}</div>", unsafe_allow_html=True)
        pal = [RED, "#1B120F"]; g = st.columns(2)
        if a["task"] == "classification":
            g[0].plotly_chart(px.bar(x=list(a["class_dist"]), y=list(a["class_dist"].values()), title="Class distribution", color_discrete_sequence=pal), use_container_width=True)
            g[1].plotly_chart(px.imshow(m["confusion"], x=a["_cm_classes"], y=a["_cm_classes"], text_auto=True, title="Confusion matrix (rows=true)", color_continuous_scale="Reds"), use_container_width=True)
            g[0].plotly_chart(px.bar(x=list(m["per_class_recall"]), y=list(m["per_class_recall"].values()), title="Per-class recall", color_discrete_sequence=pal), use_container_width=True)
        g[1].plotly_chart(px.line(x=[f"Fold {i+1}" for i in range(5)], y=a["cv"]["scores"], markers=True, title=f"Cross-validation ({a['cv']['metric']})", color_discrete_sequence=pal), use_container_width=True)
        g[0].plotly_chart(px.bar(x=["Train", "Test"], y=[m["train_score"], m[mk]], title="Train vs test", color_discrete_sequence=pal), use_container_width=True)
        g[1].plotly_chart(px.bar(x=list(a["models"]), y=[v[mk] for v in a["models"].values()], title="Model vs baseline", color_discrete_sequence=pal), use_container_width=True)
        st.button("DISCOVER BETTER DATASETS →", on_click=go_to, args=("Dataset Discovery",))

elif page == "Findings":
    st.title("Key findings")
    if not need_audit():
        for f in S.audit["findings"]:
            st.markdown(f"<div class='card'><span class='badge' style='background:{SEVC[f['severity']]}'>{f['severity']}</span> <b>{f['title']}</b><br><b>Evidence:</b> {f['evidence']}<br><b>Why it matters:</b> {f['why']}<br><b>Recommendation:</b> {f['recommendation']}</div>", unsafe_allow_html=True)

elif page == "Dataset Discovery":
    st.title("Dataset Discovery & Recommendations")
    if not need_audit():
        a = S.audit; st.subheader("CURRENT DATASET WEAKNESSES")
        for f in a["findings"]:
            if f["severity"] != "INFO": st.write(f"• **{f['title']}** — {f['evidence']}")
        st.subheader("WHAT A BETTER DATASET SHOULD HAVE")
        for n in needs_from(a): st.write(f"• {n}")
        st.subheader("RECOMMENDED DATASETS")
        st.caption("Recommended based on the detected weaknesses and available dataset characteristics. Catalog: bundled scikit-learn datasets; extendable via core/discovery.py.")
        if not S.recs: st.info("No catalog dataset matches this task type.")
        for i, r in enumerate(S.recs or []):
            c = r["info"]
            st.markdown(f"<div class='card'><b style='font-size:1.2rem'>{c['name']}</b><br><span class='lab'>{c['source'].upper()}</span><br>{c['rows']:,} rows · {c['features']} features · target '{c['target']}'" + (f" · {c['classes']} classes, minority {c['minority_pct']:.1f}%" if "classes" in c else "") + f" · missing {c['missing_pct']:.1f}%<br><br><b>Why recommended</b><br>" + "<br>".join(r["why"]) + "<br><br><b>Potential limitations</b><br>" + "<br>".join(r["limits"]) + "</div>", unsafe_allow_html=True)
            b = st.columns(2)
            if b[0].button("COMPARE", key=f"c{i}"): S.cmp = c["name"]; S.setdefault("compared", set()).add(c["name"])
            if b[1].button("AUDIT THIS DATASET", key=f"a{i}"):
                df, t = load(c["name"]); 
                if run(df, c["name"], t, c["task"]): go_to("Evaluation"); st.rerun()
        if S.get("cmp") and S.cmp in S.audits | {}:
            pass
        if S.get("cmp"):
            c = next(r["info"] for r in S.recs if r["info"]["name"] == S.cmp); P = a["profile"]
            st.subheader(f"CURRENT vs {c['name']}")
            st.table(pd.DataFrame({"Current": [f"{P['rows']:,}", str(P["cols"] - 1), f"{a.get('minority_pct', float('nan')):.1f}%", f"{P['missing_pct']:.1f}%", str(P["duplicates"])],
                S.cmp: [f"{c['rows']:,}", str(c["features"]), f"{c.get('minority_pct', float('nan')):.1f}%", f"{c['missing_pct']:.1f}%", f"{c['duplicate_pct']:.2f}% of rows"]},
                index=["Rows", "Features", "Minority share", "Missing", "Duplicates"]))
            if S.cmp in S.audits: o = S.audits[S.cmp]; st.write(f"**Re-audit result:** {o['status']} · {o['claim_conclusion']}")

elif page == "Reports":
    st.title("Reports")
    if S.audit is None and S.get("actual") is None: need_audit()
    if S.audit is not None:
        st.download_button("Download PDF report", to_pdf(S.audit, S.recs), "mlproof_report.pdf", "application/pdf")
        st.download_button("Download JSON", to_json(S.audit), "mlproof_audit.json", "application/json")
    if S.get("actual"):
        st.markdown("<div class='lab' style='margin-top:14px'>ACTUAL MODEL AUDIT</div>", unsafe_allow_html=True)
        st.download_button("Download Actual Model Audit PDF", actual_to_pdf(S.actual), "mlproof_actual_model_report.pdf", "application/pdf")
        st.download_button("Download Actual Model Audit JSON", to_json(S.actual), "mlproof_actual_model_audit.json", "application/json")
