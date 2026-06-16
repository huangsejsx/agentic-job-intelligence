from dataclasses import dataclass, asdict
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from parse_jd import parse_jd
from parse_resume import parse_resume
from match_skills import calculate_skill_match

@dataclass
class SemanticMatch:
    jd_skill: str
    best_resume_evidence: str
    similarity: float
    is_semantic_match: bool

@dataclass
class SemanticMatchResult:
    keyword_matched_skills: List[str]
    keyword_missing_skills: List[str]
    semantic_matches: List[SemanticMatch]
    final_matched_skills: List[str]
    final_missing_skills: List[str]
    keyword_score: float
    semantic_score: float

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

def semantic_match_jd_resume(jd_info, resume_info, threshold: float = 0.45) -> SemanticMatchResult:
    keyword_result = calculate_skill_match(jd_info.required_skills, resume_info.skills)
    missing_skills = keyword_result.missing_skills
    resume_evidence = build_resume_evidence(resume_info)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    evidence_embeddings = model.encode(resume_evidence, normalize_embeddings=True)
    semantic_matches = []
    semantic_matched_skills = []
    for skill in missing_skills:
        query = f"Job requirement: {skill}"
        query_embedding = model.encode(query, normalize_embeddings=True)
        similarities = [cosine_similarity(query_embedding, emb) for emb in evidence_embeddings]
        best_idx = int(np.argmax(similarities))
        best_score = round(float(similarities[best_idx]), 4)
        is_match = best_score >= threshold
        semantic_matches.append(
            SemanticMatch(
                jd_skill=skill,
                best_resume_evidence=resume_evidence[best_idx],
                similarity=best_score,
                is_semantic_match=is_match
            )
        )
        if is_match:
            semantic_matched_skills.append(skill)
    final_matched = sorted(set(keyword_result.matched_skills + semantic_matched_skills))
    final_missing = sorted(set(jd_info.required_skills) - set(final_matched))
    semantic_score = len(final_matched) / len(jd_info.required_skills) if jd_info.required_skills else 0.0
    return SemanticMatchResult(
        keyword_matched_skills=keyword_result.matched_skills,
        keyword_missing_skills=keyword_result.missing_skills,
        semantic_matches=semantic_matches,
        final_matched_skills=final_matched,
        final_missing_skills=final_missing,
        keyword_score=keyword_result.skill_match_score,
        semantic_score=round(semantic_score, 4)
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