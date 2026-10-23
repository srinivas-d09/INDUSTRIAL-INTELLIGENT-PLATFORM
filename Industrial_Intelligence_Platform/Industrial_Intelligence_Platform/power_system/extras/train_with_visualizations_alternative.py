"""Power System Load Type Classification — training + full visualizations.

Trains an ExtraTreesClassifier using a chronological holdout
(last calendar month = test), then generates 10 diagnostic figures.

Run from the project root:
    python train_with_visualizations.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="talk")
FIG_DIR = "figures"
ART_DIR = "artifacts"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)

DATA = "data/load_data.csv"
TARGET = "Load_Type"
CLASS_ORDER = ["Light_Load", "Medium_Load", "Maximum_Load"]
PALETTE = {
    "Light_Load": "#4C72B0",
    "Medium_Load": "#DD8452",
    "Maximum_Load": "#C44E52",
}


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def fig1_class_distribution(df):
    counts = df[TARGET].value_counts().reindex(CLASS_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = [PALETTE[c] for c in counts.index]

    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="black")
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val,
                     f"{val:,}\n({val/len(df)*100:.1f}%)",
                     ha="center", va="bottom", fontsize=11)
    axes[0].set_title("Class Counts", fontweight="bold")
    axes[0].set_ylabel("Samples")
    axes[0].set_ylim(0, counts.max() * 1.25)

    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=colors, startangle=90,
                wedgeprops={"edgecolor": "white"})
    axes[1].set_title("Class Share", fontweight="bold")

    plt.suptitle("Figure 1 — Class Distribution", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/01_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  OK 01_class_distribution.png")


def fig2_time_series(df):
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    axes[0].plot(df["Date_Time"], df["Usage_kWh"],
                 color="lightgray", lw=0.4, alpha=0.6, zorder=1)
    for label, color in PALETTE.items():
        sub = df[df[TARGET] == label]
        axes[0].scatter(sub["Date_Time"], sub["Usage_kWh"],
                        s=4, color=color, label=label, alpha=0.6, zorder=2)
    axes[0].set_title("Figure 2a — Usage_kWh Over Time by Load_Type", fontweight="bold")
    axes[0].set_ylabel("Usage (kWh)")
    axes[0].legend(loc="upper right", markerscale=4)

    df = df.copy()
    df["_date"] = df["Date_Time"].dt.date
    daily = df.groupby(["_date", TARGET])["Usage_kWh"].mean().unstack()
    for label, color in PALETTE.items():
        if label in daily.columns:
            axes[1].plot(daily.index, daily[label], color=color,
                         linewidth=1.6, label=label)
    axes[1].set_title("Figure 2b — Daily Mean Usage_kWh by Class", fontweight="bold")
    axes[1].set_ylabel("Mean Usage (kWh)")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/02_time_series.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  OK 02_time_series.png")


def fig3_feature_distributions(df):
    numeric = [
        "Usage_kWh",
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
        "NSM",
    ]
    numeric = [c for c in numeric if c in df.columns]

    fig, axes = plt.subplots(4, 2, figsize=(18, 18))
    axes = axes.flatten()

    for i, col in enumerate(numeric):
        sns.kdeplot(data=df, x=col, hue=TARGET, ax=axes[i],
                    hue_order=CLASS_ORDER, palette=PALETTE,
                    fill=True, common_norm=False, alpha=0.4)
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel("")

    for j in range(len(numeric), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Figure 3 — Feature Distributions by Load_Type",
                 fontsize=18, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/03_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  OK 03_feature_distributions.png")


def fig4_correlation(df):
    drop_cols = ["hour", "weekday", "month", "dayofyear", "hour_int"]
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=drop_cols, errors="ignore")
    corr = numeric_df.corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title("Figure 4 — Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  OK 04_correlation_heatmap.png")


def fig5_hourly_weekday(df):
    df = df.copy()
    df["hour_int"] = df["Date_Time"].dt.hour

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    pivot_hour = pd.crosstab(df["hour_int"], df[TARGET], normalize="index")
    pivot_hour = pivot_hour.reindex(columns=CLASS_ORDER)
    sns.heatmap(pivot_hour, ax=axes[0, 0], cmap="YlOrRd",
                cbar_kws={"label": "Proportion"})
    axes[0, 0].set_title("Load_Type Share by Hour", fontweight="bold")
    axes[0, 0].set_xlabel("Load Type")
    axes[0, 0].set_ylabel("Hour of Day")

    hourly = df.groupby(["hour_int", TARGET])["Usage_kWh"].mean().unstack()
    for label in CLASS_ORDER:
        if label in hourly.columns:
            axes[0, 1].plot(hourly.index, hourly[label], marker="o",
                            label=label, color=PALETTE[label], linewidth=2)
    axes[0, 1].set_title("Mean Usage_kWh by Hour", fontweight="bold")
    axes[0, 1].set_xlabel("Hour of Day")
    axes[0, 1].set_ylabel("Mean Usage (kWh)")
    axes[0, 1].legend()

    pivot_wd = pd.crosstab(df["weekday"], df[TARGET], normalize="index")
    pivot_wd = pivot_wd.reindex(columns=CLASS_ORDER)
    sns.heatmap(pivot_wd, ax=axes[1, 0], cmap="YlGnBu",
                cbar_kws={"label": "Proportion"})
    axes[1, 0].set_title("Load_Type Share by Weekday", fontweight="bold")
    axes[1, 0].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    axes[1, 0].set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)
    axes[1, 0].set_xlabel("Load Type")

    weekly = df.groupby(["weekday", TARGET])["Usage_kWh"].mean().unstack()
    for label in CLASS_ORDER:
        if label in weekly.columns:
            axes[1, 1].plot(weekly.index, weekly[label], marker="s",
                            label=label, color=PALETTE[label], linewidth=2)
    axes[1, 1].set_title("Mean Usage_kWh by Weekday", fontweight="bold")
    axes[1, 1].set_xticks(range(7))
    axes[1, 1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[1, 1].set_ylabel("Mean Usage (kWh)")
    axes[1, 1].legend()

    plt.suptitle("Figure 5 — Temporal Patterns by Load_Type",
                 fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/05_hourly_patterns.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  OK 05_hourly_patterns.png")


def fig6_boxplots(df):
    cols = [
        "Usage_kWh",
        "Lagging_Current_Reactive.Power_kVarh",
        "CO2(tCO2)",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
        "NSM",
    ]
    cols = [c for c in cols if c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.boxplot(data=df, x=TARGET, y=col, order=CLASS_ORDER,
                    palette=PALETTE, ax=axes[i], showfliers=False,
                    hue=TARGET, legend=False)
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel("")

    for j in range(len(cols), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Figure 6 — Feature Box Plots by Load_Type",
                 fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/06_weekday_patterns.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  OK 06_weekday_patterns.png")


def fig7_feature_importance(model, feature_names):
    importances = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=True)

    plt.figure(figsize=(11, 8))
    colors = sns.color_palette("viridis", len(importances))
    plt.barh(importances.index, importances.values,
             color=colors, edgecolor="black")
    for i, v in enumerate(importances.values):
        plt.text(v, i, f"  {v:.3f}", va="center", fontsize=10)
    plt.title("Figure 7 — Feature Importance (Extra Trees)", fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/07_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  OK 07_feature_importance.png")


def fig8_confusion_counts(y_test, pred):
    cm = confusion_matrix(y_test, pred, labels=CLASS_ORDER)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
                cbar_kws={"label": "Count"}, linewidths=0.5,
                annot_kws={"size": 14})
    plt.title("Figure 8 — Confusion Matrix (Counts)", fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/08_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  OK 08_confusion_matrix.png")


def fig9_per_class_metrics(y_test, pred):
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    metrics = ["precision", "recall", "f1-score"]
    data = {m: [report[c][m] for c in CLASS_ORDER] for m in metrics}

    x = np.arange(len(CLASS_ORDER))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, m in enumerate(metrics):
        bars = ax.bar(x + i * width, data[m], width,
                      label=m.capitalize(),
                      color=sns.color_palette("Set2")[i],
                      edgecolor="black")
        for bar, v in zip(bars, data[m]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Figure 9 — Per-Class Precision / Recall / F1", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/09_per_class_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  OK 09_per_class_metrics.png")


def fig10_confusion_normalized(y_test, pred):
    cm = confusion_matrix(y_test, pred, labels=CLASS_ORDER, normalize="true")

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Greens",
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
                cbar_kws={"label": "Row proportion"},
                linewidths=0.5, annot_kws={"size": 14})
    plt.title("Figure 10 — Confusion Matrix (Row-Normalized)", fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/10_confusion_matrix_normalized.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  OK 10_confusion_matrix_normalized.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    df = pd.read_csv(DATA)
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date_Time", TARGET]).sort_values("Date_Time").copy()

    dt = df["Date_Time"]
    df["hour"] = dt.dt.hour + dt.dt.minute / 60
    df["day"] = dt.dt.day
    df["weekday"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["dayofyear"] = dt.dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    print(f"   Rows: {len(df):,}")
    print(f"   Date range: {df['Date_Time'].min()} -> {df['Date_Time'].max()}")

    print("\nGenerating EDA figures...")
    fig1_class_distribution(df)
    fig2_time_series(df)
    fig3_feature_distributions(df)
    fig4_correlation(df)
    fig5_hourly_weekday(df)
    fig6_boxplots(df)

    last_month = df["Date_Time"].dt.to_period("M").max()
    test_mask = df["Date_Time"].dt.to_period("M") == last_month

    X = df.drop(columns=["Date_Time", TARGET])
    y = df[TARGET]

    X_train, X_test = X.loc[~test_mask], X.loc[test_mask]
    y_train, y_test = y.loc[~test_mask], y.loc[test_mask]

    print(f"\nTrain rows: {len(X_train):,}  |  Test rows: {len(X_test):,}  |  Test month: {last_month}")

    print("\nTraining Extra Trees classifier...")
    model = ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=CLASS_ORDER).tolist()

    print(f"\nAccuracy: {acc:.4f}\n")
    print(classification_report(y_test, pred, zero_division=0))
    print(f"Confusion matrix (order: {CLASS_ORDER}):")
    print(np.array(cm))

    print("\nGenerating evaluation figures...")
    fig7_feature_importance(model, list(X.columns))
    fig8_confusion_counts(y_test, pred)
    fig9_per_class_metrics(y_test, pred)
    fig10_confusion_normalized(y_test, pred)

    print("\nSaving artifacts...")
    joblib.dump(
        {"model": model, "feature_columns": list(X.columns), "test_month": str(last_month)},
        f"{ART_DIR}/load_type_model.joblib",
    )

    metrics = {
        "test_month": str(last_month),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(acc),
        "classification_report": report,
        "confusion_matrix_labels": CLASS_ORDER,
        "confusion_matrix": cm,
    }
    with open(f"{ART_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "Date_Time": df.loc[test_mask, "Date_Time"],
        "Actual_Load_Type": y_test,
        "Predicted_Load_Type": pred,
    }).to_csv(f"{ART_DIR}/last_month_predictions.csv", index=False)

    print(f"   - {ART_DIR}/load_type_model.joblib")
    print(f"   - {ART_DIR}/metrics.json")
    print(f"   - {ART_DIR}/last_month_predictions.csv")
    print(f"\nDone. 10 figures saved to ./{FIG_DIR}/")


if __name__ == "__main__":
    main()