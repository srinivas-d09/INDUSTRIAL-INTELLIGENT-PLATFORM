from pathlib import Path
import json
import subprocess
import sys
import tempfile
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "document_intelligence" / "src"))
from extractor import MetadataExtractor

st.set_page_config(page_title="Industrial Intelligence Platform", page_icon="⚙️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.hero {padding: 1.35rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#102033,#24536b); color:#fff; margin-bottom:1rem;}
.hero h1 {margin:0; font-size:2.15rem;}
.hero p {margin:.45rem 0 0; color:#d8e8f0;}
.small-note {color:#64748b; font-size:.9rem;}
</style>
<div class="hero"><h1>⚙️ Industrial Intelligence Platform</h1><p>One application for industrial load classification and contract document intelligence</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Platform Modules")
    st.caption("Two independent AI modules under one dashboard")
    st.markdown("- **Load Classification**\n  Telemetry CSV → ML prediction")
    st.markdown("- **Document Intelligence**\n  Contract file → extracted metadata")
    st.divider()
    st.caption("Evaluation metrics appear only after running on your actual dataset.")

tabs = st.tabs(["Overview", "Power System Load Classification", "Document Intelligence"])

with tabs[0]:
    st.subheader("Unified workflow")
    image_path = ROOT / "visualizations" / "platform_architecture.png"
    if image_path.exists(): st.image(str(image_path), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("AI modules", "2")
    c2.metric("Shared interface", "1 dashboard")
    c3.metric("Data connection", "Independent inputs")
    st.info("The modules are integrated through a common Streamlit interface. They solve different problems and do not share model predictions unless you add a business rule or shared identifier.")
    st.markdown("**Technology stack**")
    st.write("Python · Pandas · Scikit-learn · Transformers QA · Streamlit · Plotly · Joblib")

with tabs[1]:
    st.subheader("Power System Load Type Classification")
    st.write("Upload the telemetry CSV. The training script uses the latest calendar month as a holdout test set.")
    data_file = st.file_uploader("Upload telemetry CSV", type=["csv"], key="telemetry_csv")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        train_clicked = st.button("Train and evaluate model", type="primary", disabled=data_file is None, use_container_width=True)
    with col_b:
        show_existing = st.button("Load existing evaluation", disabled=not (ROOT / "power_system" / "artifacts" / "metrics.json").exists(), use_container_width=True)
    power_root = ROOT / "power_system"
    data_dir = power_root / "data"
    artifacts = power_root / "artifacts"
    figures = power_root / "figures"
    data_dir.mkdir(exist_ok=True)
    if train_clicked and data_file is not None:
        csv_path = data_dir / "load_data.csv"
        csv_path.write_bytes(data_file.getvalue())
        command = [sys.executable, str(power_root / "train_model.py"), "--data", str(csv_path), "--artifacts", str(artifacts), "--figures", str(figures)]
        with st.spinner("Training model and evaluating on the latest-month holdout. Large datasets may take a while..."):
            proc = subprocess.run(command, cwd=str(power_root), capture_output=True, text=True)
        if proc.returncode == 0:
            st.success("Training and holdout evaluation completed.")
            with st.expander("Training log"):
                st.code(proc.stdout[-12000:] or "Training finished.")
        else:
            st.error("Training failed. Check that the CSV contains a date/time column and a Load_Type target, with at least two months of data.")
            with st.expander("Error details"):
                st.code((proc.stderr or proc.stdout)[-12000:])
    metrics_path = artifacts / "metrics.json"
    model_path = artifacts / "load_type_model.joblib"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.markdown("### Latest-month holdout results")
            st.caption(f"Holdout month: {metrics.get('test_month','unknown')} · Train rows: {metrics.get('train_rows','—')} · Test rows: {metrics.get('test_rows','—')}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
            m2.metric("Balanced accuracy", f"{metrics.get('balanced_accuracy', 0):.1%}")
            m3.metric("Macro recall", f"{metrics.get('macro_recall', 0):.1%}")
            m4.metric("Macro F1", f"{metrics.get('macro_f1', 0):.1%}")
            labels = metrics.get("labels", [])
            cm = metrics.get("confusion_matrix", [])
            if labels and cm:
                cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                fig = px.imshow(cm_df, text_auto=True, aspect="auto", title="Holdout confusion matrix", labels=dict(x="Predicted", y="Actual", color="Count"))
                st.plotly_chart(fig, use_container_width=True)
            report = metrics.get("classification_report", {})
            class_rows = []
            for label in labels:
                if label in report:
                    class_rows.append({"Load type": label, "Precision": report[label].get("precision", 0), "Recall": report[label].get("recall", 0), "F1-score": report[label].get("f1-score", 0), "Support": report[label].get("support", 0)})
            if class_rows:
                st.dataframe(pd.DataFrame(class_rows).style.format({"Precision":"{:.3f}","Recall":"{:.3f}","F1-score":"{:.3f}"}), hide_index=True, use_container_width=True)
            pred_path = artifacts / "last_month_predictions.csv"
            if pred_path.exists():
                st.download_button("Download holdout predictions CSV", pred_path.read_bytes(), file_name="last_month_predictions.csv", mime="text/csv")
        except Exception as e:
            st.warning(f"Could not read existing metrics: {e}")
    else:
        st.info("No verified model metrics are available yet. Upload the original dataset and click **Train and evaluate model** to generate them.")
    if model_path.exists():
        st.markdown("### Predict from a telemetry CSV")
        pred_file = st.file_uploader("Upload telemetry rows to classify", type=["csv"], key="predict_csv")
        if pred_file is not None and st.button("Predict load types", key="predict_btn"):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("load_train_module", power_root / "train_model.py")
                module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
                bundle = joblib.load(model_path)
                df = module.normalize_columns(pd.read_csv(pred_file))
                time_col = next((c for c in module.TIME_COL_CANDIDATES if c in df.columns), None)
                if time_col is None: raise ValueError("CSV needs a date/time column.")
                if time_col != "Date_Time": df = df.rename(columns={time_col:"Date_Time"})
                df["Date_Time"] = pd.to_datetime(df["Date_Time"], dayfirst=True, errors="coerce")
                engineered = module.engineer_features(df)
                X = engineered.reindex(columns=bundle["feature_columns"])
                X = X.replace([float("inf"), float("-inf")], float("nan")).fillna(pd.Series(bundle["medians"])).fillna(0)
                pred = bundle["model"].predict(X)
                out = df.copy(); out["Predicted_Load_Type"] = pred
                st.dataframe(out.head(100), use_container_width=True)
                st.download_button("Download predictions", out.to_csv(index=False), file_name="load_type_predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    st.caption("The benchmark is not prefilled. Metrics are computed from the uploaded data and trained model.")

with tabs[2]:
    st.subheader("Document Metadata Extraction")
    st.write("Upload a rental agreement. The QA model attempts to extract six fields from the document text.")
    doc_file = st.file_uploader("Upload DOCX, PDF, or image", type=["docx", "pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"], key="contract_doc")
    st.caption("Scanned PDFs/images may require Tesseract OCR installed separately. The first run downloads the pretrained QA model.")
    if st.button("Extract contract metadata", type="primary", disabled=doc_file is None, use_container_width=False):
        temp_dir = ROOT / "outputs" / "uploaded_documents"
        temp_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(doc_file.name).name
        temp_path = temp_dir / safe_name
        temp_path.write_bytes(doc_file.getvalue())
        try:
            with st.spinner("Loading QA model and extracting fields..."):
                extractor = MetadataExtractor()
                result = extractor.extract_file(temp_path)
            st.session_state["metadata_result"] = result
            st.success("Extraction finished. Review values before relying on them.")
        except Exception as e:
            st.error(f"Extraction failed: {e}")
    result = st.session_state.get("metadata_result")
    if result:
        labels = {"agreement_value":"Agreement Value", "agreement_start_date":"Agreement Start Date", "agreement_end_date":"Agreement End Date", "renewal_notice_days":"Renewal Notice (Days)", "party_one":"Party One", "party_two":"Party Two"}
        rows = [{"Field": label, "Extracted value": result["fields"][key]["value"] or "Not confidently extracted", "Confidence": float(result["fields"][key]["confidence"])} for key, label in labels.items()]
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)
        fig = px.bar(df, x="Field", y="Confidence", range_y=[0,1], title="QA model confidence by field")
        fig.update_layout(xaxis_tickangle=-25, margin=dict(l=8,r=8,t=45,b=8))
        st.plotly_chart(fig, use_container_width=True)
        export = pd.DataFrame([{"File Name": result.get("file_name", doc_file.name if doc_file else ""), **{key: result["fields"][key]["value"] for key in labels}}])
        st.download_button("Download extracted metadata CSV", export.to_csv(index=False), file_name="metadata_predictions.csv", mime="text/csv")
        st.warning("Confidence is not a guarantee of correctness. Verify legal and financial fields manually.")
    else:
        st.info("Extracted values and confidence chart will appear here after you upload a contract and run extraction.")

st.divider()
st.caption("Integrated interface demo. Add the real datasets and run the workflows before reporting any benchmark or recall metrics.")
