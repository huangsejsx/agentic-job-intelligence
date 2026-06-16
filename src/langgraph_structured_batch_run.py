from pathlib import Path
from agent_workflow_structured import run_structured_workflow

def langgraph_structured_batch_run(jd_dir: str, resume_path: str, output_dir: str) -> None:
    jd_files = sorted(Path(jd_dir).glob("*.txt"))
    if not jd_files:
        print(f"No JD files found in {jd_dir}")
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for jd_file in jd_files:
        output_path = Path(output_dir) / f"{jd_file.stem}_decision.json"
        print(f"Analyzing {jd_file}...")
        final_state = run_structured_workflow(
            jd_path=str(jd_file),
            resume_path=resume_path,
            output_path=str(output_path)
        )
        jd_info = final_state["jd_info"]
        decision = final_state["decision"]
        print(f"Title: {jd_info.get('title')}")
        print(f"Company: {jd_info.get('company')}")
        print(f"Location: {jd_info.get('location')}")
        print(f"Decision: {decision['decision']}")
        print(f"Keyword score: {decision.get('keyword_score')}")
        print(f"Semantic score: {decision.get('semantic_score')}")
        print(f"Trace: {' -> '.join(final_state['trace'])}")
        print("-" * 50)

if __name__ == "__main__":
    langgraph_structured_batch_run(
        jd_dir="data/job_descriptions",
        resume_path="data/resumes/my_resume.txt",
        output_dir="results/langgraph_structured_batch_decisions"
    )