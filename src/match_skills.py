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

@dataclass
class WeightedSkillMatchResult:
    required_result: SkillMatchResult
    preferred_result: SkillMatchResult
    weighted_score: float
    required_weight: float
    preferred_weight: float

def combine_required_preferred_scores(
    required_score: float,
    preferred_score: float,
    has_required_skills: bool,
    has_preferred_skills: bool,
    required_weight: float = 0.8
) -> float:
    preferred_weight = 1.0 - required_weight
    if has_required_skills and has_preferred_skills:
        return round(
            required_weight * required_score + preferred_weight * preferred_score,
            4
        )
    if has_required_skills:
        return round(required_score, 4)
    if has_preferred_skills:
        return round(preferred_score, 4)
    return 0.0

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

def calculate_weighted_skill_match(
    required_skills: List[str],
    preferred_skills: List[str],
    resume_skills: List[str],
    required_weight: float = 0.8
) -> WeightedSkillMatchResult:
    preferred_weight = round(1.0 - required_weight, 4)
    required_result = calculate_skill_match(required_skills, resume_skills)
    preferred_result = calculate_skill_match(preferred_skills, resume_skills)
    weighted_score = combine_required_preferred_scores(
        required_result.skill_match_score,
        preferred_result.skill_match_score,
        bool(required_skills),
        bool(preferred_skills),
        required_weight
    )
    return WeightedSkillMatchResult(
        required_result=required_result,
        preferred_result=preferred_result,
        weighted_score=weighted_score,
        required_weight=required_weight,
        preferred_weight=preferred_weight
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
