import re
from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class JDInfo:
    title: Optional[str]
    company: Optional[str]
    required_skills: List[str]
    preferred_skills: List[str]
    education_requirements: List[str]
    experience_years: Optional[int]
    requires_phd: bool
    requires_many_years: bool
    hard_filter_reason: Optional[str]

SKILL_PATTERNS = {
    "python": [r"\bpython\b"],
    "sql": [r"\bsql\b"],
    "machine learning": [r"\bmachine learning\b", r"\bml\b"],
    "deep learning": [r"\bdeep learning\b", r"\bdl\b"],
    "statistics": [r"\bstatistics\b", r"\bstatistical\b"],
    "data analysis": [r"\bdata analysis\b", r"\banalytics\b"],
    "data science": [r"\bdata science\b"],
    "pandas": [r"\bpandas\b"],
    "numpy": [r"\bnumpy\b"],
    "scikit-learn": [r"\bscikit-learn\b", r"\bsklearn\b"],
    "tensorflow": [r"\btensorflow\b"],
    "pytorch": [r"\bpytorch\b"],
    "keras": [r"\bkeras\b"],
    "nlp": [r"\bnlp\b", r"\bnatural language processing\b"],
    "llm": [r"\bllm\b", r"\blarge language model"],
    "rag": [r"\brag\b", r"\bretrieval augmented generation\b"],
    "agent": [r"\bagent\b", r"\bagentic\b"],
    "langchain": [r"\blangchain\b"],
    "spark": [r"\bspark\b", r"\bpyspark\b"],
    "hadoop": [r"\bhadoop\b"],
    "aws": [r"\baws\b", r"\bamazon web services\b"],
    "azure": [r"\bazure\b"],
    "gcp": [r"\bgcp\b", r"\bgoogle cloud\b"],
    "tableau": [r"\btableau\b"],
    "power bi": [r"\bpower bi\b"],
    "excel": [r"\bexcel\b"],
    "ab testing": [r"\ba/b testing\b", r"\ba/b tests?\b", r"\bab testing\b", r"\bab tests?\b", r"\ba-b testing\b", r"\ba-b tests?\b", r"\bexperiment design\b"],
    "recommendation": [r"\brecommendation\b", r"\brecommender\b", r"\brecommender systems?\b"],
    "ranking": [r"\branking\b", r"\brankings\b"],
    "search": [r"\bsearch\b"],
    "retrieval": [r"\bretrieval\b"],
    "data visualization": [r"\bvisualization\b", r"\bvisualisation\b", r"\bdata viz\b", r"\bdashboard"]
}

def extract_title(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else None

def extract_company(text: str) -> Optional[str]:
    patterns = [
        r"Company:\s*(.+)",
        r"at\s+([A-Z][A-Za-z0-9&.\-\s]+)",
        r"About\s+([A-Z][A-Za-z0-9&.\-\s]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None

def extract_skills(text: str) -> List[str]:
    lower_text = text.lower()
    skills = []
    for skill, patterns in SKILL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lower_text):
                skills.append(skill)
                break
    return sorted(set(skills))

def extract_education(text: str) -> List[str]:
    lower_text = text.lower()
    education = []
    keywords = ["bachelor", "master", "msc", "phd", "doctorate", "degree"]
    for keyword in keywords:
        if keyword in lower_text:
            education.append(keyword)
    return sorted(set(education))

def extract_experience_years(text: str) -> Optional[int]:
    patterns = [
        r"(\d+)\+?\s*years?",
        r"(\d+)\+?\s*yrs?"
    ]
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        years.extend([int(x) for x in matches])
    return max(years) if years else None

def check_hard_filters(text: str, experience_years: Optional[int]) -> tuple[bool, bool, Optional[str]]:
    lower_text = text.lower()
    requires_phd = any(word in lower_text for word in ["phd", "doctorate", "doctoral"])
    requires_many_years = experience_years is not None and experience_years >= 3
    if requires_phd:
        return requires_phd, requires_many_years, "Requires PhD or doctoral-level qualification."
    if requires_many_years:
        return requires_phd, requires_many_years, f"Requires {experience_years}+ years of experience."
    return requires_phd, requires_many_years, None

def parse_jd(text: str) -> JDInfo:
    title = extract_title(text)
    company = extract_company(text)
    skills = extract_skills(text)
    education = extract_education(text)
    experience_years = extract_experience_years(text)
    requires_phd, requires_many_years, hard_filter_reason = check_hard_filters(text, experience_years)
    return JDInfo(
        title=title,
        company=company,
        required_skills=skills,
        preferred_skills=[],
        education_requirements=education,
        experience_years=experience_years,
        requires_phd=requires_phd,
        requires_many_years=requires_many_years,
        hard_filter_reason=hard_filter_reason
    )

if __name__ == "__main__":
    sample_jd = """
    Data Science Intern
    Company: ExampleTech

    We are looking for a Data Science Intern with strong Python, SQL,
    statistics, machine learning, pandas, and scikit-learn skills.
    Experience with recommendation systems, ranking, and A/B testing is preferred.
    Candidates should be pursuing a Bachelor's or Master's degree.
    """

    info = parse_jd(sample_jd)
    print(asdict(info))