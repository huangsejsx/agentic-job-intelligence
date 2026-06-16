import json
import pandas as pd
from pathlib import Path

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_keyword_results(keyword_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(keyword_dir).glob("*.json")):
        data = load_json(path)
        decision = data["decision"]
        rows.append({
            "job_file": Path(data["jd_path"]).name,
            "keyword_decision": decision["decision"],
            "keyword_score": decision.get("skill_match_score"),
            "keyword_hard_filter_triggered": decision["hard_filter_triggered"],
            "keyword_trace": " -> ".join(data.get("trace", []))
        })
    return pd.DataFrame(rows)

def load_semantic_results(semantic_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(semantic_dir).glob("*.json")):
        data = load_json(path)
        decision = data["decision"]
        semantic_matches = []
        for item in decision.get("semantic_evidence", []):
            if item.get("is_semantic_match"):
                semantic_matches.append(
                    f"{item['jd_skill']} -> {item['best_resume_evidence']} ({item['similarity']})"
                )
        rows.append({
            "job_file": Path(data["jd_path"]).name,
            "semantic_decision": decision["decision"],
            "semantic_score": decision.get("semantic_score"),
            "semantic_hard_filter_triggered": decision["hard_filter_triggered"],
            "semantic_trace": " -> ".join(data.get("trace", [])),
            "semantic_matches": " | ".join(semantic_matches),
            "final_missing_skills": ", ".join(decision.get("final_missing_skills", []))
        })
    return pd.DataFrame(rows)

def compare_keyword_semantic(
    keyword_dir: str,
    semantic_dir: str,
    label_path: str,
    output_path: str
) -> None:
    keyword_df = load_keyword_results(keyword_dir)
    semantic_df = load_semantic_results(semantic_dir)
    label_df = pd.read_csv(label_path)
    df = label_df.merge(keyword_df, on="job_file", how="left")
    df = df.merge(semantic_df, on="job_file", how="left")
    if df["keyword_decision"].isna().any():
        missing = df[df["keyword_decision"].isna()]["job_file"].tolist()
        raise ValueError(f"Missing keyword predictions for: {missing}")
    if df["semantic_decision"].isna().any():
        missing = df[df["semantic_decision"].isna()]["job_file"].tolist()
        raise ValueError(f"Missing semantic predictions for: {missing}")
    df["decision_changed"] = df["keyword_decision"] != df["semantic_decision"]
    df["keyword_correct"] = df["keyword_decision"] == df["true_label"]
    df["semantic_correct"] = df["semantic_decision"] == df["true_label"]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved comparison to {output_path}")
    print(df[[
        "job_file",
        "true_label",
        "keyword_decision",
        "keyword_score",
        "semantic_decision",
        "semantic_score",
        "decision_changed",
        "keyword_correct",
        "semantic_correct"
    ]])

if __name__ == "__main__":
    compare_keyword_semantic(
        keyword_dir="results/langgraph_batch_decisions",
        semantic_dir="results/langgraph_semantic_batch_decisions",
        label_path="data/labels/job_labels.csv",
        output_path="results/keyword_vs_semantic_comparison.csv"
    )