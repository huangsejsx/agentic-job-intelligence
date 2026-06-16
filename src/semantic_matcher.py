from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from parse_jd import parse_jd
from parse_resume import parse_resume
from match_skills import calculate_skill_match, combine_required_preferred_scores

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

@dataclass
class SemanticMatch:
    jd_skill: str
    skill_type: str
    best_resume_evidence: str
    similarity: float
    is_semantic_match: bool

@dataclass
class SemanticMatchResult:
    keyword_matched_skills: List[str]
    keyword_missing_skills: List[str]
    preferred_keyword_matched_skills: List[str]
    preferred_keyword_missing_skills: List[str]
    semantic_matches: List[SemanticMatch]
    final_matched_skills: List[str]
    final_missing_skills: List[str]
    final_preferred_matched_skills: List[str]
    final_preferred_missing_skills: List[str]
    keyword_score: float
    semantic_score: float
    required_keyword_score: float
    preferred_keyword_score: float
    required_semantic_score: float
    preferred_semantic_score: float
    required_weight: float
    preferred_weight: float

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)

def build_resume_evidence(resume_info) -> List[str]:
    evidence = []
    for skill in resume_info.skills:
        evidence.append(f"Skill: {skill}")
    for edu in resume_info.education:
        evidence.append(f"Education: {edu}")
    for project in resume_info.projects:
        evidence.append(f"Project: {project}")
    for exp in resume_info.experiences:
        evidence.append(f"Experience: {exp}")
    return evidence

@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def semantic_match_skill_group(
    skills: List[str],
    keyword_matched_skills: List[str],
    keyword_missing_skills: List[str],
    resume_evidence: List[str],
    evidence_embeddings,
    model,
    threshold: float,
    skill_type: str
) -> tuple[List[SemanticMatch], List[str], List[str], float]:
    semantic_matches = []
    semantic_matched_skills = []
    if resume_evidence and model is not None:
        skills_to_semantically_match = keyword_missing_skills
    else:
        skills_to_semantically_match = []
    for skill in skills_to_semantically_match:
        query = f"Job requirement: {skill}"
        query_embedding = model.encode(query, normalize_embeddings=True)
        similarities = [cosine_similarity(query_embedding, emb) for emb in evidence_embeddings]
        best_idx = int(np.argmax(similarities))
        best_score = round(float(similarities[best_idx]), 4)
        is_match = best_score >= threshold
        semantic_matches.append(
            SemanticMatch(
                jd_skill=skill,
                skill_type=skill_type,
                best_resume_evidence=resume_evidence[best_idx],
                similarity=best_score,
                is_semantic_match=is_match
            )
        )
        if is_match:
            semantic_matched_skills.append(skill)
    final_matched = sorted(set(keyword_matched_skills + semantic_matched_skills))
    final_missing = sorted(set(skills) - set(final_matched))
    semantic_score = len(final_matched) / len(skills) if skills else 0.0
    return semantic_matches, final_matched, final_missing, round(semantic_score, 4)

def semantic_match_jd_resume(
    jd_info,
    resume_info,
    threshold: float = 0.45,
    required_weight: float = 0.8
) -> SemanticMatchResult:
    required_keyword_result = calculate_skill_match(jd_info.required_skills, resume_info.skills)
    preferred_keyword_result = calculate_skill_match(jd_info.preferred_skills, resume_info.skills)
    resume_evidence = build_resume_evidence(resume_info)
    missing_skills = required_keyword_result.missing_skills + preferred_keyword_result.missing_skills
    if resume_evidence and missing_skills:
        model = get_embedding_model()
        evidence_embeddings = model.encode(resume_evidence, normalize_embeddings=True)
    else:
        model = None
        evidence_embeddings = []
    required_semantic_matches, final_required_matched, final_required_missing, required_semantic_score = semantic_match_skill_group(
        jd_info.required_skills,
        required_keyword_result.matched_skills,
        required_keyword_result.missing_skills,
        resume_evidence,
        evidence_embeddings,
        model,
        threshold,
        "required"
    )
    preferred_semantic_matches, final_preferred_matched, final_preferred_missing, preferred_semantic_score = semantic_match_skill_group(
        jd_info.preferred_skills,
        preferred_keyword_result.matched_skills,
        preferred_keyword_result.missing_skills,
        resume_evidence,
        evidence_embeddings,
        model,
        threshold,
        "preferred"
    )
    semantic_score = combine_required_preferred_scores(
        required_semantic_score,
        preferred_semantic_score,
        bool(jd_info.required_skills),
        bool(jd_info.preferred_skills),
        required_weight
    )
    keyword_score = combine_required_preferred_scores(
        required_keyword_result.skill_match_score,
        preferred_keyword_result.skill_match_score,
        bool(jd_info.required_skills),
        bool(jd_info.preferred_skills),
        required_weight
    )
    return SemanticMatchResult(
        keyword_matched_skills=required_keyword_result.matched_skills,
        keyword_missing_skills=required_keyword_result.missing_skills,
        preferred_keyword_matched_skills=preferred_keyword_result.matched_skills,
        preferred_keyword_missing_skills=preferred_keyword_result.missing_skills,
        semantic_matches=required_semantic_matches + preferred_semantic_matches,
        final_matched_skills=final_required_matched,
        final_missing_skills=final_required_missing,
        final_preferred_matched_skills=final_preferred_matched,
        final_preferred_missing_skills=final_preferred_missing,
        keyword_score=keyword_score,
        semantic_score=semantic_score,
        required_keyword_score=required_keyword_result.skill_match_score,
        preferred_keyword_score=preferred_keyword_result.skill_match_score,
        required_semantic_score=required_semantic_score,
        preferred_semantic_score=preferred_semantic_score,
        required_weight=required_weight,
        preferred_weight=round(1.0 - required_weight, 4)
    )

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    jd_text = read_text("data/job_descriptions/sample_jd_1.txt")
    resume_text = read_text("data/resumes/my_resume.txt")
    jd_info = parse_jd(jd_text)
    resume_info = parse_resume(resume_text)
    result = semantic_match_jd_resume(jd_info, resume_info)
    print(asdict(result))
