import json
import pandas as pd
from pathlib import Path

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def summarize_langgraph_decisions(input_dir: str, output_path: str) -> None:
    rows = []
    json_files = sorted(Path(input_dir).glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    for file_path in json_files:
        data = load_json(file_path)
        jd_info = data["jd_info"]
        decision = data["decision"]
        trace = data.get("trace", [])
        rows.append({
            "job_file": Path(data["jd_path"]).name,
            "result_file": file_path.name,
            "title": jd_info["title"],
            "company": jd_info["company"],
            "decision": decision["decision"],
            "final_score": decision["final_score"],
            "skill_match_score": decision["skill_match_score"],
            "hard_filter_triggered": decision["hard_filter_triggered"],
            "hard_filter_reason": decision["hard_filter_reason"],
            "matched_skills": ", ".join(decision["matched_skills"]),
            "missing_skills": ", ".join(decision["missing_skills"]),
            "trace": " -> ".join(trace),
            "explanation": decision["explanation"]
        })
    df = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved LangGraph summary to {output_path}")
    print(df[["job_file", "title", "company", "decision", "skill_match_score", "hard_filter_triggered", "trace"]])

if __name__ == "__main__":
    summarize_langgraph_decisions(
        input_dir="results/langgraph_batch_decisions",
        output_path="results/langgraph_batch_summary.csv"
    )