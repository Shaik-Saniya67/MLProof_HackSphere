"""Summary report (PDF / JSON) for the Model 1 workflow. Kept separate from report_generator.py."""
import io, json
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def to_json(summary):
    return json.dumps(summary, indent=2, default=str)


def _table(rows, header, widths=None):
    data = [header] + rows
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F4EDDD")), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#CDBFA5")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def to_pdf(s):
    buf = io.BytesIO(); st = getSampleStyleSheet(); S = []
    P = lambda t, style="BodyText": S.append(Paragraph(escape(str(t)), st[style]))
    H = lambda t: (S.append(Spacer(1, 8)), S.append(Paragraph(escape(t), st["Heading2"])))
    cell = lambda v: Paragraph(escape(str(v)), st["BodyText"]) if len(str(v)) > 28 else str(v)
    f = s["findings"]
    S.append(Paragraph("MLProof - Dataset Analysis &amp; Model Comparison", st["Title"]))
    P(f"Dataset: {s['dataset']}. Models trained on: {s['trained_on']}.")
    H("1. Dataset findings (original dataset)")
    P(f"{f['rows']:,} rows and {f['columns']} columns ({f['numeric_columns']} numeric, {f['categorical_columns']} categorical).")
    P(f"Missing cells: {f['missing_cells']:,} (rows with at least one missing value: {f['rows_with_missing']:,}). Exact duplicate rows: {f['duplicate_rows']:,}.")
    P(f"Constant columns: {', '.join(f['constant_columns']) or 'none'}. Possible ID-like columns: {', '.join(f['id_like_columns']) or 'none'}. "
      f"Columns with IQR outliers: {', '.join(f['columns_with_outliers']) or 'none'}.")
    H("2. Cleaning actions")
    if s["cleaning"]:
        rows = [[cell(c["Action"]), cell(c["Detail"]), c["Rows removed"], c["Columns removed"]] for c in s["cleaning"]]
        S.append(_table(rows, ["Action", "Detail", "Rows removed", "Columns removed"], [50 * mm, 80 * mm, 22 * mm, 22 * mm]))
        if s.get("cleaned_shape"):
            P(f"Cleaned dataset size: {s['cleaned_shape'][0]:,} rows x {s['cleaned_shape'][1]} columns. The original data was not modified.")
    else:
        P("No cleaning was applied. The original dataset was used.")
    m = s.get("model")
    if m:
        H("3. Model settings")
        P(f"Task: {m['task']}. Target: {m['target']}. Features used: {m['n_features']} ({', '.join(m['features'][:20])}{'...' if len(m['features']) > 20 else ''}).")
        P(f"Train rows: {m['train_rows']:,}. Test rows: {m['test_rows']:,}. Test size: {m['settings']['test_size']:.0%}. Random seed: {m['settings']['seed']}.")
        P(f"Decision tree max depth: {m['settings']['tree_depth']}. Random forest trees: {m['settings']['rf_trees']}, max depth: {m['settings']['rf_depth'] or 'unlimited'}. "
          f"Missing numeric values imputed with the {m['settings']['impute_num']}; missing categories with {m['settings']['impute_cat'].replace('_', ' ')}. "
          "Imputation, scaling and encoding were fitted on the training split only.")
        used = set(m["models_trained"])
        extra = [t for t, n, k in (("XGBoost trees", "XGBoost", "xgb_trees"), ("CatBoost iterations", "CatBoost", "cat_iters"), ("TabNet max epochs", "TabNet", "tabnet_epochs")) if n in used and k in m["settings"]]
        if extra:
            keys = {"XGBoost trees": "xgb_trees", "CatBoost iterations": "cat_iters", "TabNet max epochs": "tabnet_epochs"}
            P("Boosting / deep model settings: " + ", ".join(f"{t} {m['settings'][keys[t]]}" for t in extra) + ". TabNet early stopping used a slice of the training rows only.")
        if m.get("averaging"):
            P(f"Precision, recall and F1 use the {m['averaging']}.")
        H("4. Evaluation metrics (held-out test set)")
        cols = list(s["metrics"][0].keys())
        S.append(_table([[str(r[c]) for c in cols] for r in s["metrics"]], cols))
        P(f"Best non-baseline model by {m['best_by']}: {m['best_model']}.")
        if m["models_failed"]:
            P("Models that failed: " + "; ".join(f"{k}: {v}" for k, v in m["models_failed"].items()))
        H("5. Interpretation")
        for t in s["explanation"]:
            P(t)
        for t in s["warnings"]:
            P("Warning: " + t)
        for t in m["notes"]:
            P("Note: " + t)
    H("Notes")
    P("These results describe one evaluation on one train/test split. They are not guarantees of future performance. Outliers are reported, not removed.")
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm).build(S)
    return buf.getvalue()
