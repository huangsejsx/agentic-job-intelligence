import json
import pandas as pd
from pathlib import Path

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_list(items: list) -> str:
    return " | ".join(items) if items else ""

def format_semantic_evidence(evidence: list) -> str:
    matched_items = []
    for item in evidence:
        if item.get("is_semantic_match"):
            matched_items.append(
                f"{item['jd_skill']} -> {item['best_resume_evidence']} ({item['similarity']})"
            )
    return " | ".join(matched_items)

def summarize_structured_decisions(input_dir: str, output_path: str) -> None:
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
            "title": jd_info.get("title"),
            "company": jd_info.get("company"),
            "location": jd_info.get("location"),
            "decision": decision["decision"],
            "final_score": decision["final_score"],
            "keyword_score": decision.get("keyword_score"),
            "semantic_score": decision.get("semantic_score"),
            "hard_filter_triggered": decision["hard_filter_triggered"],
            "hard_filter_reason": decision["hard_filter_reason"],
            "required_skills": ", ".join(jd_info.get("required_skills", [])),
            "responsibilities": format_list(jd_info.get("responsibilities", [])),
            "keyword_missing_skills": ", ".join(decision.get("keyword_missing_skills", [])),
            "final_missing_skills": ", ".join(decision.get("final_missing_skills", [])),
            "semantic_evidence": format_semantic_evidence(decision.get("semantic_evidence", [])),
            "trace": " -> ".join(trace),
            "extraction_source": jd_info.get("extraction_source"),
            "explanation": decision["explanation"]
        })
    df = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved structured summary to {output_path}")
    print(df[[
        "job_file",
        "title",
        "company",
        "location",
        "decision",
        "keyword_score",
        "semantic_score",
        "hard_filter_triggered",
        "trace"
    ]])

if __name__ == "__main__":
    summarize_structured_decisions(
        input_dir="results/langgraph_structured_batch_decisions",
        output_path="results/langgraph_structured_batch_summary.csv"
    )