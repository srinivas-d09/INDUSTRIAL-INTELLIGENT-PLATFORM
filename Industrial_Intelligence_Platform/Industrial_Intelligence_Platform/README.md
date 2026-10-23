# Industrial Intelligence Platform

A unified Streamlit interface that brings together two separate AI/ML modules:

1. **Power System Load Type Classification** — trains an Extra Trees classifier and evaluates it using the latest calendar month as a holdout set.
2. **Document Intelligence** — uses a pretrained extractive question-answering transformer to extract six fields from rental agreements.

The modules share a dashboard, but they solve different tasks. No claim is made that one model improves the other.

## Project structure

```text
Industrial_Intelligence_Platform/
├── app.py                         # unified dashboard
├── requirements.txt
├── README.md
├── visualizations/
│   └── platform_architecture.png  # workflow diagram, not performance data
├── power_system/                  # original load classification project
│   ├── app.py
│   ├── train_model.py
│   ├── load_type_classification.ipynb
│   ├── requirements.txt
│   ├── Problem_Statement.docx
│   ├── data/README.txt
│   ├── artifacts/                 # generated after training
│   └── figures/                   # generated after training
├── document_intelligence/         # original document extraction project
│   ├── app.py
│   ├── api.py
│   ├── requirements.txt
│   ├── src/extractor.py
│   ├── src/evaluate.py
│   └── src/predict_test.py
├── data/
└── outputs/
```

## Setup (Windows PowerShell)

Python 3.11 is recommended, especially for the Transformers/PyTorch stack.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens with three tabs: Overview, Power System Load Type Classification, and Document Intelligence.

## Power system module

Upload the assignment's original telemetry CSV in the Load Classification tab. The CSV must include a date/time column and `Load_Type` target. At least two calendar months are required because the latest calendar month is held out for testing. Click **Train and evaluate model**. The app runs `power_system/train_model.py`, saves the model/metrics/predictions, and visualizes holdout metrics and the confusion matrix.

For command-line training instead:

```powershell
python power_system/train_model.py --data "power_system/data/load_data.csv"
```

Expected outputs: `power_system/artifacts/metrics.json`, `load_type_model.joblib`, `last_month_predictions.csv`, and figures in `power_system/figures/`.

## Document intelligence module

Upload a DOCX, PDF, or supported image in the Document Intelligence tab. The app uses `deepset/roberta-base-squad2` for question-answering extraction. The first run downloads the model and requires internet access. For scanned PDFs or images, install Tesseract OCR separately and add it to PATH.

The six target fields are agreement value, start date, end date, renewal notice days, party one, and party two. Verify extracted values manually.

## Important dataset and evaluation notes

- The original load telemetry CSV was not included in the provided project files. Add the real assignment dataset before training.
- The rental agreement training/test CSVs and documents were not included in the provided files. Add them before running predictions/evaluation.
- This ZIP contains **no invented performance metrics**. Load-model metrics appear only after training on the actual CSV. Document extraction confidence is not accuracy or recall.
- `visualizations/platform_architecture.png` depicts the software workflow; it does not represent model performance.

## Separate original apps

The original module apps are preserved at `power_system/app.py` and `document_intelligence/app.py`, along with their training/extraction source files. The root `app.py` is the combined dashboard.
