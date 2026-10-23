"""Compute exact-match recall for each metadata field."""
import argparse
import re
import pandas as pd

FIELD_KEYS = [
    "agreement_value", "agreement_start_date", "agreement_end_date",
    "renewal_notice_days", "party_one", "party_two"
]

def norm(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()

def find_column(df, candidates):
    normalized = {re.sub(r"[^a-z0-9]", "", str(c).casefold()): c for c in df.columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.casefold())
        if key in normalized:
            return normalized[key]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/test.csv")
    parser.add_argument("--predictions", default="outputs/test_predictions.csv")
    parser.add_argument("--out", default="outputs/recall_report.csv")
    args = parser.parse_args()
    labels = pd.read_csv(args.labels)
    preds = pd.read_csv(args.predictions)
    label_name = find_column(labels, ["File Name", "file_name", "filename"])
    pred_name = find_column(preds, ["File Name", "file_name", "filename"])
    if not label_name or not pred_name:
        raise SystemExit("Both CSV files need a file-name column.")
    merged = labels.merge(preds, left_on=label_name, right_on=pred_name, suffixes=("_label", "_pred"))
    report = []
    for key in FIELD_KEYS:
        label_col = find_column(merged, [key + "_label", key])
        pred_col = find_column(merged, [key + "_pred", key])
        # Resolve duplicate column names explicitly after merge.
        if key + "_label" in merged.columns: label_col = key + "_label"
        if key + "_pred" in merged.columns: pred_col = key + "_pred"
        if not label_col or not pred_col:
            report.append({"Field": key, "Exact Matches": 0, "Labeled Examples": 0, "Recall": 0.0})
            continue
        valid = merged[label_col].map(norm) != ""
        count = int(valid.sum())
        matches = int((merged.loc[valid, label_col].map(norm) == merged.loc[valid, pred_col].map(norm)).sum()) if count else 0
        report.append({"Field": key, "Exact Matches": matches, "Labeled Examples": count,
                       "Recall": matches / count if count else 0.0})
    out = pd.DataFrame(report)
    out_path = __import__("pathlib").Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"Saved report to {out_path}")

if __name__ == "__main__":
    main()
