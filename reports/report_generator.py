import io, json
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

def to_pdf(a, recs=None):
    buf = io.BytesIO(); st = getSampleStyleSheet(); S = []; P = a["profile"]
    h = lambda t: S.extend([Spacer(1, 10), Paragraph(t, st["Heading2"])]); p = lambda t: S.append(Paragraph(str(t).replace("<", "&lt;"), st["BodyText"]))
    S.append(Paragraph("MLProof - Evaluation Integrity Report", st["Title"]))
    h("1. Executive Summary"); p(f"Status: {a['status']}"); p(a["claim_conclusion"])
    h("2. Dataset Profile"); p(f"{P['rows']:,} rows, {P['cols']} columns, target '{a['target']}', task: {a['task']}.")
    h("3. Data Quality"); p(f"Missing: {P['missing_pct']:.1f}%. Duplicates: {P['duplicates']}. Constant: {P['constant']}. ID-like: {P['id_cols']}.")
    if "class_dist" in a: h("4. Class Distribution"); p(f"{a['class_dist']} (ratio {a['imbalance_ratio']:.1f}:1)")
    h("5. Leakage & Contamination"); p(f"Train/test overlapping records: {a['overlap']}. Leakage items are heuristics, not confirmed leakage.")
    h("6. Model Performance")
    for n, m in a["models"].items(): p(n + ": " + ", ".join(f"{k}={v:.3f}" for k, v in m.items() if isinstance(v, float)))
    h("7. Validation Stability"); c = a["cv"]; p(f"Folds: {[round(s, 3) for s in c['scores']]}, mean {c['mean']:.3f}, std {c['std']:.3f}, range {c['range']:.3f}")
    h("8. Baseline Comparison"); p(f"Improvement over baseline: {a['improvement']*100:+.1f} points")
    h("9. Key Findings")
    for f in a["findings"]: p(f"[{f['severity']}] {f['title']} - {f['evidence']} Why: {f['why']} Fix: {f['recommendation']}")
    h("10-12. Dataset Recommendations & Comparison")
    for r in (recs or [])[:3]: p(f"{r['info']['name']} ({r['info']['source']}): " + "; ".join(r["why"]))
    p("Recommended based on the detected weaknesses and available dataset characteristics.")
    h("13. Final Evaluation Status"); p(a["status"])
    SimpleDocTemplate(buf, pagesize=A4).build(S); return buf.getvalue()

def to_json(a): return json.dumps({k: v for k, v in a.items() if not k.startswith("_")}, indent=2, default=str)

def actual_to_pdf(a):
    buf = io.BytesIO(); st = getSampleStyleSheet(); S = []
    h = lambda t: S.extend([Spacer(1, 10), Paragraph(t, st["Heading2"])]); p = lambda t: S.append(Paragraph(str(t).replace("<", "&lt;"), st["BodyText"]))
    S.append(Paragraph("MLProof - Actual Model Audit Report", st["Title"]))
    h("1. Executive Summary"); p(f"Status: {a['status']}"); p(a["conclusion"]); p(a["scope"])
    h("2. Evaluation Data"); p(f"{a['n_used']:,} of {a['n_rows']:,} rows used. Predictions: '{a['pred_col']}', ground truth: '{a['true_col']}', task: {a['task']}.")
    h("3. Performance Claim"); c = a["claim"]; p(f"Claim: {c['text'] or 'none supplied'}")
    if c.get("result"): p(f"Claimed {c['result']['claimed']:.4g}, recomputed {c['result']['computed']:.4g}.")
    elif c.get("note"): p(c["note"])
    h("4. Recomputed Metrics"); p(", ".join(f"{k}={v:.4f}" for k, v in a["metrics"].items()))
    if "per_class" in a:
        h("5. Per-Class Metrics")
        for k, v in a["per_class"].items(): p(f"{k}: precision {v['precision']:.3f}, recall {v['recall']:.3f}, F1 {v['f1']:.3f}, support {v['support']}")
    h("6. Key Findings")
    for f in a["findings"]: p(f"[{f['severity']}] {f['title']} - {f['evidence']} Why: {f['why']} Recommendation: {f['recommendation']}")
    h("7. Final Evaluation Status"); p(a["status"])
    SimpleDocTemplate(buf, pagesize=A4).build(S); return buf.getvalue()
