from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from parse_jd import parse_jd, JDInfo
from parse_resume import parse_resume, ResumeInfo
from semantic_matcher import semantic_match_jd_resume

@dataclass
class SemanticDecisionResult:
    decision: str
    final_score: float
    hard_filter_triggered: bool
    hard_filter_reason: Optional[str]
    keyword_score: float
    semantic_score: float
    keyword_matched_skills: List[str]
    keyword_missing_skills: List[str]
    final_matched_skills: List[str]
    final_missing_skills: List[str]
    semantic_evidence: List[Dict[str, Any]]
    explanation: str

def make_semantic_decision(jd_info: JDInfo, resume_info: ResumeInfo) -> SemanticDecisionResult:
    if jd_info.requires_phd or jd_info.requires_many_years:
        return SemanticDecisionResult(
            decision="Pass",
            final_score=0.0,
            hard_filter_triggered=True,
            hard_filter_reason=jd_info.hard_filter_reason,
            keyword_score=0.0,
            semantic_score=0.0,
            keyword_matched_skills=[],
            keyword_missing_skills=jd_info.required_skills,
            final_matched_skills=[],
            final_missing_skills=jd_info.required_skills,
            semantic_evidence=[],
            explanation=f"Pass because the role has a hard requirement: {jd_info.hard_filter_reason}"
        )
    match_result = semantic_match_jd_resume(jd_info, resume_info)
    semantic_evidence = []
    for item in match_result.semantic_matches:
        semantic_evidence.append({
            "jd_skill": item.jd_skill,
            "best_resume_evidence": item.best_resume_evidence,
            "similarity": item.similarity,
            "is_semantic_match": item.is_semantic_match
        })
    score = match_result.semantic_score
    if score >= 0.85:
        decision = "Apply"
        explanation = "Apply because the calibrated semantic match score is high and no hard filter is triggered."
    elif score >= 0.45:
        decision = "Maybe"
        explanation = "Maybe because the role has partial semantic match, but the calibrated score is not high enough for Apply."
    else:
        decision = "Pass"
        explanation = "Pass because the semantic match score is low."
    return SemanticDecisionResult(
        decision=decision,
        final_score=score,
        hard_filter_triggered=False,
        hard_filter_reason=None,
        keyword_score=match_result.keyword_score,
        semantic_score=match_result.semantic_score,
        keyword_matched_skills=match_result.keyword_matched_skills,
        keyword_missing_skills=match_result.keyword_missing_skills,
        final_matched_skills=match_result.final_matched_skills,
        final_missing_skills=match_result.final_missing_skills,
        semantic_evidence=semantic_evidence,
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
    result = make_semantic_decision(jd_info, resume_info)
    print(asdict(result))