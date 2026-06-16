from pathlib import Path
from run_pipeline import run_single_job_analysis

def batch_run(jd_dir: str, resume_path: str, output_dir: str) -> None:
    jd_files = sorted(Path(jd_dir).glob("*.txt"))
    if not jd_files:
        print(f"No JD files found in {jd_dir}")
        return
    for jd_file in jd_files:
        output_path = Path(output_dir) / f"{jd_file.stem}_decision.json"
        print(f"Analyzing {jd_file}...")
        run_single_job_analysis(
            jd_path=str(jd_file),
            resume_path=resume_path,
            output_path=str(output_path)
        )
        print("-" * 50)

if __name__ == "__main__":
    batch_run(
        jd_dir="data/job_descriptions",
        resume_path="data/resumes/my_resume.txt",
        output_dir="results/batch_decisions"
    )