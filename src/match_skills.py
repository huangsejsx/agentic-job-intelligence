from dataclasses import dataclass, asdict
from typing import List
from parse_jd import parse_jd
from parse_resume import parse_resume

@dataclass
class SkillMatchResult:
    jd_skills: List[str]
    resume_skills: List[str]
    matched_skills: List[str]
    missing_skills: List[str]
    skill_match_score: float

def calculate_skill_match(jd_skills: List[str], resume_skills: List[str]) -> SkillMatchResult:
    jd_set = set(jd_skills)
    resume_set = set(resume_skills)
    matched = sorted(jd_set & resume_set)
    missing = sorted(jd_set - resume_set)
    score = len(matched) / len(jd_set) if jd_set else 0.0
    return SkillMatchResult(
        jd_skills=sorted(jd_set),
        resume_skills=sorted(resume_set),
        matched_skills=matched,
        missing_skills=missing,
        skill_match_score=round(score, 4)
    )

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    jd_text = read_text("data/job_descriptions/sample_jd_1.txt")
    resume_text = read_text("data/resumes/my_resume.txt")
    jd_info = parse_jd(jd_text)
    resume_info = parse_resume(resume_text)
    result = calculate_skill_match(jd_info.required_skills, resume_info.skills)
    print(asdict(result))