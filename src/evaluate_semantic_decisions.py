import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_semantic_predictions(results_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        data = load_json(path)
        job_file = Path(data["jd_path"]).name
        decision = data["decision"]
        trace = " -> ".join(data.get("trace", []))
        rows.append({
            "job_file": job_file,
            "pred_label": decision["decision"],
            "keyword_score": decision.get("keyword_score"),
            "semantic_score": decision.get("semantic_score"),
            "hard_filter_triggered": decision["hard_filter_triggered"],
            "trace": trace
        })
    return pd.DataFrame(rows)

def evaluate_semantic(results_dir: str, label_path: str, output_path: str) -> None:
    pred_df = load_semantic_predictions(results_dir)
    label_df = pd.read_csv(label_path)
    df = label_df.merge(pred_df, on="job_file", how="left")
    if df["pred_label"].isna().any():
        missing = df[df["pred_label"].isna()]["job_file"].tolist()
        raise ValueError(f"Missing predictions for: {missing}")
    labels = ["Apply", "Maybe", "Pass"]
    y_true = df["true_label"]
    y_pred = df["pred_label"]
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    metrics = pd.DataFrame([{
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "num_samples": len(df)
    }])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)
    df.to_csv("results/semantic_evaluation_predictions.csv", index=False)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels]
    )
    cm_df.to_csv("results/semantic_confusion_matrix.csv")
    with open("results/semantic_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("Semantic evaluation results:")
    print(metrics)
    print("\nPredictions:")
    print(df)
    print("\nConfusion matrix:")
    print(cm_df)
    print("\nClassification report:")
    print(report)

if __name__ == "__main__":
    evaluate_semantic(
        results_dir="results/langgraph_semantic_batch_decisions",
        label_path="data/labels/job_labels.csv",
        output_path="results/semantic_evaluation_metrics.csv"
    )