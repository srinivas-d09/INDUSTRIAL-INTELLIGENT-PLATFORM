"""Run metadata extraction on documents listed in data/test.csv."""
import argparse
from pathlib import Path
import pandas as pd
from extractor import MetadataExtractor

FIELD_KEYS = [
    "agreement_value", "agreement_start_date", "agreement_end_date",
    "renewal_notice_days", "party_one", "party_two"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="outputs/test_predictions.csv")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    labels_path = data_dir / "test.csv"
    if not labels_path.exists():
        raise SystemExit(f"Missing {labels_path}. Add the assignment's test.csv and test documents.")
    labels = pd.read_csv(labels_path)
    if "File Name" not in labels.columns and "file_name" not in labels.columns:
        raise SystemExit("test.csv must include a file-name column (File Name or file_name).")
    file_col = "File Name" if "File Name" in labels.columns else "file_name"
    extractor = MetadataExtractor()
    rows = []
    for _, row in labels.iterrows():
        name = str(row[file_col])
        candidates = [data_dir / "test" / name, data_dir / "test" / f"{name}.docx",
                      data_dir / "test" / f"{name}.png", data_dir / "test" / f"{name}.pdf"]
        file_path = next((p for p in candidates if p.is_file()), None)
        if file_path is None:
            matches = list((data_dir / "test").glob(name + ".*"))
            file_path = matches[0] if matches else None
        record = {"File Name": name}
        if file_path:
            result = extractor.extract_file(file_path)
            record.update({k: result["fields"][k]["value"] for k in FIELD_KEYS})
        else:
            record.update({k: "" for k in FIELD_KEYS})
            record["Error"] = "Document file not found in data/test/"
        rows.append(record)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved predictions to {out}")

if __name__ == "__main__":
    main()
