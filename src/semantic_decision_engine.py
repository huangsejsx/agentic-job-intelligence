from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from parse_jd import parse_jd, JDInfo
from parse_resume import parse_resume, ResumeInfo
from semantic_matcher import semantic_match_jd_resume

APPLY_THRESHOLD = 0.85
MAYBE_THRESHOLD = 0.45
REQUIRED_STRONG_MATCH_THRESHOLD = 0.95
REQUIRED_APPLY_GATE_THRESHOLD = 0.90

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
    final_preferred_matched_skills: List[str]
    final_preferred_missing_skills: List[str]
    score_breakdown: Dict[str, Any]
    semantic_evidence: List[Dict[str, Any]]
    explanation: str

def build_semantic_score_breakdown(match_result) -> Dict[str, Any]:
    return {
        "required_keyword_score": match_result.required_keyword_score,
        "preferred_keyword_score": match_result.preferred_keyword_score,
        "weighted_keyword_score": match_result.keyword_score,
        "required_semantic_score": match_result.required_semantic_score,
        "preferred_semantic_score": match_result.preferred_semantic_score,
        "weighted_semantic_score": match_result.semantic_score,
        "required_weight": match_result.required_weight,
        "preferred_weight": match_result.preferred_weight,
        "matched_required_skill_count": len(match_result.final_matched_skills),
        "missing_required_skill_count": len(match_result.final_missing_skills),
        "matched_preferred_skill_count": len(match_result.final_preferred_matched_skills),
        "missing_preferred_skill_count": len(match_result.final_preferred_missing_skills)
    }

def build_hard_filter_score_breakdown(jd_info: JDInfo) -> Dict[str, Any]:
    return {
        "required_keyword_score": 0.0,
        "preferred_keyword_score": 0.0,
        "weighted_keyword_score": 0.0,
        "required_semantic_score": 0.0,
        "preferred_semantic_score": 0.0,
        "weighted_semantic_score": 0.0,
        "required_weight": 0.8,
        "preferred_weight": 0.2,
        "matched_required_skill_count": 0,
        "missing_required_skill_count": len(jd_info.required_skills),
        "matched_preferred_skill_count": 0,
        "missing_preferred_skill_count": len(jd_info.preferred_skills)
    }

def choose_semantic_decision(weighted_score: float, required_score: float) -> tuple[str, str]:
    if required_score >= REQUIRED_STRONG_MATCH_THRESHOLD:
        return (
            "Apply",
            "Apply because the required skills are strongly matched; missing preferred skills do not block the decision."
        )
    if weighted_score >= APPLY_THRESHOLD and required_score >= REQUIRED_APPLY_GATE_THRESHOLD:
        return (
            "Apply",
            "Apply because both the weighted semantic score and required skill score are high."
        )
    if weighted_score >= MAYBE_THRESHOLD:
        return (
            "Maybe",
            "Maybe because the role has partial semantic match, but the required skill score is not high enough for Apply."
        )
    return (
        "Pass",
        "Pass because the semantic match score is low."
    )

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
            final_preferred_matched_skills=[],
            final_preferred_missing_skills=jd_info.preferred_skills,
            score_breakdown=build_hard_filter_score_breakdown(jd_info),
            semantic_evidence=[],
            explanation=f"Pass because the role has a hard requirement: {jd_info.hard_filter_reason}"
        )
    match_result = semantic_match_jd_resume(jd_info, resume_info)
    semantic_evidence = []
    for item in match_result.semantic_matches:
        semantic_evidence.append({
            "jd_skill": item.jd_skill,
            "skill_type": item.skill_type,
            "best_resume_evidence": item.best_resume_evidence,
            "similarity": item.similarity,
            "is_semantic_match": item.is_semantic_match
        })
    score = match_result.semantic_score
    decision, explanation = choose_semantic_decision(
        weighted_score=score,
        required_score=match_result.required_semantic_score
    )
    score_breakdown = build_semantic_score_breakdown(match_result)
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
        final_preferred_matched_skills=match_result.final_preferred_matched_skills,
        final_preferred_missing_skills=match_result.final_preferred_missing_skills,
        score_breakdown=score_breakdown,
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
