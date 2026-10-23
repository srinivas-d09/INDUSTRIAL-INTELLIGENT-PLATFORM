from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent / "src"))
from extractor import MetadataExtractor, read_document, FIELDS

st.set_page_config(page_title="DocuLens AI", page_icon="📄", layout="wide")
st.markdown("""
<style>
.main {background: #f6f8fc;}
.hero {padding: 1.25rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#111827,#243b65); color: white;}
.hero h1 {margin:0; font-size:2.1rem;}
.hero p {margin:.45rem 0 0; color:#dbeafe;}
.metric-card {padding: 1rem; border-radius: 14px; background: white; border: 1px solid #e5e7eb;}
</style>
<div class="hero"><h1>📄 DocuLens AI</h1><p>Template-independent contract metadata extraction using pretrained question answering</p></div>
""", unsafe_allow_html=True)
st.write("")
left, right = st.columns([1, 1.5], gap="large")
with left:
    st.subheader("1 · Upload a document")
    uploaded = st.file_uploader("DOCX, PDF, PNG, JPG or TIFF", type=["docx","pdf","png","jpg","jpeg","tif","tiff","bmp","webp"])
    st.caption("Scanned documents use OCR. For images/PDF scans, install Tesseract OCR on your computer.")
    run = st.button("Extract metadata", type="primary", use_container_width=True, disabled=uploaded is None)
    st.info("The first run downloads the pretrained QA model and may take a few minutes.")
with right:
    st.subheader("2 · Extracted metadata")
    if run and uploaded:
        tmp = Path("outputs") / uploaded.name
        tmp.parent.mkdir(exist_ok=True)
        tmp.write_bytes(uploaded.getvalue())
        try:
            with st.spinner("Reading document and asking the AI model..."):
                model = MetadataExtractor()
                output = model.extract_file(tmp)
            st.session_state["last_output"] = output
            st.session_state["last_name"] = uploaded.name
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.caption("Check that requirements are installed and that the model can be downloaded.")
    output = st.session_state.get("last_output")
    if output:
        st.caption(f"File: {st.session_state.get('last_name', output['file_name'])}")
        rows = []
        labels = {
            "agreement_value":"Agreement Value",
            "agreement_start_date":"Agreement Start Date",
            "agreement_end_date":"Agreement End Date",
            "renewal_notice_days":"Renewal Notice (Days)",
            "party_one":"Party One",
            "party_two":"Party Two",
        }
        for key, label in labels.items():
            item = output["fields"][key]
            rows.append({"Field": label, "Extracted value": item["value"] or "Not confidently extracted",
                         "Confidence": round(item["confidence"], 3)})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)
        fig = px.bar(df, x="Field", y="Confidence", range_y=[0,1], title="Model confidence by field")
        fig.update_layout(margin=dict(l=8,r=8,t=45,b=8), xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download results as CSV", pd.DataFrame([{
            "File Name": output["file_name"],
            **{k: output["fields"][k]["value"] for k in labels}
        }]).to_csv(index=False), file_name="metadata_predictions.csv", mime="text/csv")
    else:
        st.markdown("Upload a rental agreement and click **Extract metadata** to view the six target fields.")
st.divider()
st.subheader("How the system works")
c1,c2,c3 = st.columns(3)
c1.markdown("### 01 · Read\nDOCX text extraction or OCR for scanned images/PDFs.")
c2.markdown("### 02 · Understand\nA pretrained extractive question-answering model finds answers from document context.")
c3.markdown("### 03 · Inspect\nReview extracted values, confidence scores, and export the result.")
st.caption("Important: confidence is a model score, not a guarantee of correctness. Always verify legal/financial values.")
