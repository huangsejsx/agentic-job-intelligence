import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

def decide_by_threshold(score: float, hard_filter_triggered: bool, apply_threshold: float, maybe_threshold: float) -> str:
    if hard_filter_triggered:
        return "Pass"
    if score >= apply_threshold:
        return "Apply"
    if score >= maybe_threshold:
        return "Maybe"
    return "Pass"

def run_threshold_calibration(summary_path: str, label_path: str, output_path: str) -> None:
    summary_df = pd.read_csv(summary_path)
    label_df = pd.read_csv(label_path)
    df = label_df.merge(summary_df, on="job_file", how="left")
    if df["semantic_score"].isna().any():
        missing = df[df["semantic_score"].isna()]["job_file"].tolist()
        raise ValueError(f"Missing semantic scores for: {missing}")
    labels = ["Apply", "Maybe", "Pass"]
    apply_thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    maybe_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]
    rows = []
    for apply_th in apply_thresholds:
        for maybe_th in maybe_thresholds:
            if maybe_th >= apply_th:
                continue
            pred_labels = []
            for _, row in df.iterrows():
                pred = decide_by_threshold(
                    score=row["semantic_score"],
                    hard_filter_triggered=bool(row["hard_filter_triggered"]),
                    apply_threshold=apply_th,
                    maybe_threshold=maybe_th
                )
                pred_labels.append(pred)
            accuracy = accuracy_score(df["true_label"], pred_labels)
            macro_f1 = f1_score(df["true_label"], pred_labels, labels=labels, average="macro", zero_division=0)
            rows.append({
                "apply_threshold": apply_th,
                "maybe_threshold": maybe_th,
                "accuracy": round(accuracy, 4),
                "macro_f1": round(macro_f1, 4),
                "predictions": " | ".join([f"{job}:{pred}" for job, pred in zip(df["job_file"], pred_labels)])
            })
    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(["macro_f1", "accuracy"], ascending=False)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output, index=False)
    print(f"Saved threshold calibration results to {output_path}")
    print(result_df.head(10))

if __name__ == "__main__":
    run_threshold_calibration(
        summary_path="results/langgraph_structured_batch_summary.csv",
        label_path="data/labels/job_labels.csv",
        output_path="results/threshold_calibration_results.csv"
    )