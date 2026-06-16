from pathlib import Path
from agent_workflow_conditional import run_conditional_workflow

def langgraph_batch_run(jd_dir: str, resume_path: str, output_dir: str) -> None:
    jd_files = sorted(Path(jd_dir).glob("*.txt"))
    if not jd_files:
        print(f"No JD files found in {jd_dir}")
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for jd_file in jd_files:
        output_path = Path(output_dir) / f"{jd_file.stem}_decision.json"
        print(f"Analyzing {jd_file}...")
        final_state = run_conditional_workflow(
            jd_path=str(jd_file),
            resume_path=resume_path,
            output_path=str(output_path)
        )
        print(f"Decision: {final_state['decision']['decision']}")
        print(f"Trace: {' -> '.join(final_state['trace'])}")
        print("-" * 50)

if __name__ == "__main__":
    langgraph_batch_run(
        jd_dir="data/job_descriptions",
        resume_path="data/resumes/my_resume.txt",
        output_dir="results/langgraph_batch_decisions"
    )