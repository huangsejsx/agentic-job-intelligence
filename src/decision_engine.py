from dataclasses import dataclass, asdict
from typing import List, Optional
from parse_jd import parse_jd, JDInfo
from parse_resume import parse_resume, ResumeInfo
from match_skills import calculate_weighted_skill_match

@dataclass
class DecisionResult:
    decision: str
    final_score: float
    hard_filter_triggered: bool
    hard_filter_reason: Optional[str]
    skill_match_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    explanation: str

def make_decision(jd_info: JDInfo, resume_info: ResumeInfo) -> DecisionResult:
    weighted_result = calculate_weighted_skill_match(
        jd_info.required_skills,
        jd_info.preferred_skills,
        resume_info.skills
    )
    skill_result = weighted_result.required_result
    if jd_info.requires_phd or jd_info.requires_many_years:
        return DecisionResult(
            decision="Pass",
            final_score=0.0,
            hard_filter_triggered=True,
            hard_filter_reason=jd_info.hard_filter_reason,
            skill_match_score=skill_result.skill_match_score,
            matched_skills=skill_result.matched_skills,
            missing_skills=skill_result.missing_skills,
            explanation=f"Pass because the role has a hard requirement: {jd_info.hard_filter_reason}"
        )
    score = weighted_result.weighted_score
    if score >= 0.65:
        decision = "Apply"
        explanation = "Apply because the weighted required/preferred skill match score is high and no hard filter is triggered."
    elif score >= 0.4:
        decision = "Maybe"
        explanation = "Maybe because the role has partial weighted skill overlap but some important skills are missing."
    else:
        decision = "Pass"
        explanation = "Pass because the weighted skill match score is low."
    return DecisionResult(
        decision=decision,
        final_score=score,
        hard_filter_triggered=False,
        hard_filter_reason=None,
        skill_match_score=score,
        matched_skills=skill_result.matched_skills,
        missing_skills=skill_result.missing_skills,
        explanation=explanation
    )

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    jd_text = read_text("data/job_descriptions/sample_jd_1.txt")
    resume_text = read_text("data/resumes/my_resume.txt")
    jd_info = parse_jd(jd_text)
    resume_info = parse_resume(resume_text)
    result = make_decision(jd_info, resume_info)
    print(asdict(result))
