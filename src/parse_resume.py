import re
from dataclasses import dataclass, asdict
from typing import List, Optional

RESUME_SKILL_PATTERNS = {
    "python": [r"\bpython\b"],
    "sql": [r"\bsql\b"],
    "machine learning": [r"\bmachine learning\b", r"\bml\b"],
    "deep learning": [r"\bdeep learning\b", r"\bdl\b"],
    "statistics": [r"\bstatistics\b", r"\bstatistical\b"],
    "data analysis": [r"\bdata analysis\b", r"\banalysis\b", r"\banalytics\b"],
    "pandas": [r"\bpandas\b"],
    "numpy": [r"\bnumpy\b"],
    "scikit-learn": [r"\bscikit-learn\b", r"\bsklearn\b"],
    "pytorch": [r"\bpytorch\b"],
    "keras": [r"\bkeras\b"],
    "sql": [r"\bsql\b"],
    "ab testing": [r"\ba/b testing\b", r"\ba/b tests?\b", r"\bab testing\b", r"\bab tests?\b"],
    "recommendation": [r"\brecommendation\b", r"\brecommender\b", r"\brecommender systems?\b"],
    "ranking": [r"\branking\b"],
    "rag": [r"\brag\b", r"\bretrieval augmented generation\b"],
    "agent": [r"\bagent\b", r"\bagentic\b"],
    "langchain": [r"\blangchain\b"],
    "streamlit": [r"\bstreamlit\b"],
    "excel": [r"\bexcel\b"],
    "tableau": [r"\btableau\b"],
    "power bi": [r"\bpower bi\b"]
}

@dataclass
class ResumeInfo:
    name: Optional[str]
    education: List[str]
    skills: List[str]
    projects: List[str]
    experiences: List[str]

def extract_name(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else None

def extract_resume_skills(text: str) -> List[str]:
    lower_text = text.lower()
    skills = []
    for skill, patterns in RESUME_SKILL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lower_text):
                skills.append(skill)
                break
    return sorted(set(skills))

def extract_section(text: str, section_name: str) -> List[str]:
    pattern = rf"{section_name}:\s*(.*?)(?=\n[A-Z][A-Za-z ]+:|\Z)"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return []
    content = match.group(1).strip()
    lines = [line.strip("- ").strip() for line in content.splitlines() if line.strip()]
    return lines

def parse_resume(text: str) -> ResumeInfo:
    name = extract_name(text)
    education = extract_section(text, "Education")
    projects = extract_section(text, "Projects")
    experiences = extract_section(text, "Experience")
    skills = extract_resume_skills(text)
    return ResumeInfo(
        name=name,
        education=education,
        skills=skills,
        projects=projects,
        experiences=experiences
    )

if __name__ == "__main__":
    path = "data/resumes/my_resume.txt"
    with open(path, "r", encoding="utf-8") as f:
        resume_text = f.read()
    info = parse_resume(resume_text)
    print(asdict(info))