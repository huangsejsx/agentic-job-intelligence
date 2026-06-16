import json
from pathlib import Path
from tools import parse_jd_tool, parse_resume_tool, job_decision_tool

def save_json(data: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def run_single_job_analysis(jd_path: str, resume_path: str, output_path: str) -> None:
    jd_info = parse_jd_tool(jd_path)
    resume_info = parse_resume_tool(resume_path)
    decision = job_decision_tool(jd_path, resume_path)
    result = {
        "jd_path": jd_path,
        "resume_path": resume_path,
        "jd_info": jd_info,
        "resume_info": resume_info,
        "decision": decision
    }
    save_json(result, output_path)
    print(f"Saved analysis result to {output_path}")
    print(f"Decision: {decision['decision']}")
    print(f"Skill match score: {decision['skill_match_score']}")

if __name__ == "__main__":
    run_single_job_analysis(
        jd_path="data/job_descriptions/sample_jd_1.txt",
        resume_path="data/resumes/my_resume.txt",
        output_path="results/sample_jd_1_decision.json"
    )