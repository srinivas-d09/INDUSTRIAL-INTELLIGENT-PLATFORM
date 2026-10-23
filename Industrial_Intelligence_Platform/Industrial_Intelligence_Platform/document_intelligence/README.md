# DocuLens AI — Assignment 2: Document Metadata Extraction

## Objective
Extract six rental-agreement fields from documents with different templates:
- Agreement Value
- Agreement Start Date
- Agreement End Date
- Renewal Notice (Days)
- Party One
- Party Two

The assignment says not to use a rule-based approach (such as regular expressions or static conditions). This baseline uses a pretrained extractive question-answering transformer and asks a natural-language question for each field.

## Project structure
```text
DocuLens_AI_Assignment_2/
├── app.py                       # Streamlit interface, confidence chart, CSV export
├── api.py                       # Optional FastAPI endpoint
├── ARCHITECTURE.md
├── assignment-details.pdf
├── requirements.txt
├── src/
│   ├── extractor.py             # DOCX/PDF/image reader and QA model
│   ├── predict_test.py          # Predictions for the provided test set
│   └── evaluate.py              # Per-field exact-match recall
├── data/
│   ├── train/                   # Put supplied training documents here
│   ├── test/                    # Put supplied test documents here
│   ├── train.csv                # Add the assignment-provided labels
│   └── test.csv                 # Add the assignment-provided labels
└── outputs/                     # Generated CSVs/reports
```

**Dataset note:** the uploaded materials did not include the assignment's `data/` dataset (`train.csv`, `test.csv`, or the contract documents). These files must be added before test-set predictions and recall can be reproduced. No recall score is claimed in this package.

## Requirements
Python 3.10 or 3.11 recommended. The first run downloads `deepset/roberta-base-squad2` from Hugging Face and requires internet access.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For scanned images or scanned PDFs, install Tesseract OCR separately and make sure `tesseract` is on PATH.

## Run the Streamlit app
From the project root:
```powershell
streamlit run app.py
```
Upload a DOCX, PDF, or supported image. Review all values before using them: the model confidence is not a guarantee of correctness.

## Run the REST API (optional)
```powershell
uvicorn api:app --reload
```
- Health check: `GET http://127.0.0.1:8000/health`
- Swagger docs: `http://127.0.0.1:8000/docs`
- Extraction: `POST http://127.0.0.1:8000/extract` with multipart form field named `file`.

Example:
```bash
curl -X POST "http://127.0.0.1:8000/extract" -F "file=@agreement.docx"
```

## Generate predictions and evaluate
After adding the assignment-provided CSVs and matching files:
```powershell
python src/predict_test.py --data-dir data --out outputs/test_predictions.csv
python src/evaluate.py --labels data/test.csv --predictions outputs/test_predictions.csv
```
The recall script reports exact normalized string matches per field. Formatting differences (for example, date or currency formatting) may count as mismatches. Review the output and report the actual recall; do not report unrun or estimated scores.

## Limitations
- A general pretrained QA model may confuse agreement value with monthly rent or confuse party roles.
- OCR quality affects scanned files.
- The current reader processes the first 12,000 characters as QA context; long agreements may need chunking.
- Verify extracted legal and financial values manually.
