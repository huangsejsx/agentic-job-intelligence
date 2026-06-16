from dataclasses import asdict
from typing import Dict, Any
from parse_jd import parse_jd
from parse_resume import parse_resume
from decision_engine import make_decision
from semantic_decision_engine import make_semantic_decision
from structured_jd_extractor import extract_jd_structured

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def parse_jd_tool(jd_path: str) -> Dict[str, Any]:
    jd_text = read_text_file(jd_path)
    jd_info = parse_jd(jd_text)
    return asdict(jd_info)

def structured_jd_extraction_tool(jd_path: str) -> Dict[str, Any]:
    jd_text = read_text_file(jd_path)
    jd_info = extract_jd_structured(jd_text)
    return asdict(jd_info)

def parse_resume_tool(resume_path: str) -> Dict[str, Any]:
    resume_text = read_text_file(resume_path)
    resume_info = parse_resume(resume_text)
    return asdict(resume_info)

def job_decision_tool(jd_path: str, resume_path: str) -> Dict[str, Any]:
    jd_text = read_text_file(jd_path)
    resume_text = read_text_file(resume_path)
    jd_info = parse_jd(jd_text)
    resume_info = parse_resume(resume_text)
    decision = make_decision(jd_info, resume_info)
    return asdict(decision)

def semantic_job_decision_tool(jd_path: str, resume_path: str) -> Dict[str, Any]:
    jd_text = read_text_file(jd_path)
    resume_text = read_text_file(resume_path)
    jd_info = parse_jd(jd_text)
    resume_info = parse_resume(resume_text)
    decision = make_semantic_decision(jd_info, resume_info)
    return asdict(decision)

if __name__ == "__main__":
    simple_jd_path = "data/job_descriptions/sample_jd_1.txt"
    complex_jd_path = "data/job_descriptions/sample_jd_4_complex_format.txt"
    resume_path = "data/resumes/my_resume.txt"

    print("=== Rule-based JD Tool Output ===")
    print(parse_jd_tool(complex_jd_path))

    print("\n=== Structured JD Extraction Tool Output ===")
    print(structured_jd_extraction_tool(complex_jd_path))

    print("\n=== Resume Tool Output ===")
    print(parse_resume_tool(resume_path))

    print("\n=== Keyword Decision Tool Output ===")
    print(job_decision_tool(simple_jd_path, resume_path))

    print("\n=== Semantic Decision Tool Output ===")
    print(semantic_job_decision_tool(simple_jd_path, resume_path))
