import json
import re
from dataclasses import dataclass, asdict
from typing import List, Optional
from parse_jd import parse_jd

@dataclass
class StructuredJDInfo:
    title: Optional[str]
    company: Optional[str]
    location: Optional[str]
    required_skills: List[str]
    preferred_skills: List[str]
    education_requirements: List[str]
    experience_years: Optional[int]
    requires_phd: bool
    requires_many_years: bool
    responsibilities: List[str]
    hard_constraints: List[str]
    extraction_source: str

SECTION_HEADERS = {
    "about the job", "who we are", "about the team", "responsibilities",
    "minimum qualifications", "preferred qualifications", "requirements",
    "qualifications", "location", "what you will do", "what you'll do"
}

TITLE_KEYWORDS = [
    "intern", "analyst", "scientist", "engineer", "associate", "manager",
    "developer", "researcher", "consultant", "specialist"
]

COMPANY_PATTERNS = [
    r"Company:\s*(.+)",
    r"\b([A-Z][A-Za-z0-9&.\-]+)\s+is\s+(?:a|an)\s+",
    r"\b([A-Z][A-Za-z0-9&.\-]+)\s+is\s+looking\s+for",
    r"\bAt\s+([A-Z][A-Za-z0-9&.\-]+)",
]

def clean_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]

def extract_title_rule(text: str) -> Optional[str]:
    lines = clean_lines(text)
    for line in lines[:20]:
        lower_line = line.lower()
        if lower_line in SECTION_HEADERS:
            continue
        if any(keyword in lower_line for keyword in TITLE_KEYWORDS):
            return line
    return lines[0] if lines else None

def extract_company_rule(text: str) -> Optional[str]:
    for pattern in COMPANY_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None

def extract_location_rule(text: str) -> Optional[str]:
    patterns = [
        r"Location:\s*(.+)",
        r"based in\s+([A-Z][A-Za-z,\s]+)",
        r"located in\s+([A-Z][A-Za-z,\s]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    lines = clean_lines(text)
    for i, line in enumerate(lines):
        if line.lower() == "location" and i + 1 < len(lines):
            return lines[i + 1].strip()
    return None

def get_section_lines(text: str, start_headers: List[str], stop_headers: List[str]) -> List[str]:
    lines = clean_lines(text)
    collecting = False
    section_lines = []
    start_set = {x.lower() for x in start_headers}
    stop_set = {x.lower() for x in stop_headers}
    for line in lines:
        lower_line = line.lower()
        if lower_line in start_set:
            collecting = True
            continue
        if collecting and lower_line in stop_set:
            break
        if collecting:
            section_lines.append(line.strip("-• ").strip())
    return section_lines

def extract_responsibilities_rule(text: str) -> List[str]:
    responsibilities = get_section_lines(
        text,
        start_headers=["Responsibilities", "What you will do", "What you'll do"],
        stop_headers=["Minimum Qualifications", "Preferred Qualifications", "Requirements", "Qualifications", "Location"]
    )
    if responsibilities:
        return responsibilities[:10]
    lines = [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]
    responsibility_keywords = [
        "analyze", "build", "design", "evaluate", "work with",
        "develop", "create", "support", "monitor", "define", "communicate"
    ]
    extracted = []
    for line in lines:
        lower_line = line.lower()
        if any(keyword in lower_line for keyword in responsibility_keywords):
            extracted.append(line)
    return extracted[:10]

def extract_jd_with_rules(text: str) -> StructuredJDInfo:
    rule_info = parse_jd(text)
    title = extract_title_rule(text) or rule_info.title
    company = extract_company_rule(text) or rule_info.company
    location = extract_location_rule(text)
    responsibilities = extract_responsibilities_rule(text)
    hard_constraints = []
    if rule_info.requires_phd:
        hard_constraints.append("Requires PhD or doctoral-level qualification.")
    if rule_info.requires_many_years:
        hard_constraints.append(f"Requires {rule_info.experience_years}+ years of experience.")
    return StructuredJDInfo(
        title=title,
        company=company,
        location=location,
        required_skills=rule_info.required_skills,
        preferred_skills=rule_info.preferred_skills,
        education_requirements=rule_info.education_requirements,
        experience_years=rule_info.experience_years,
        requires_phd=rule_info.requires_phd,
        requires_many_years=rule_info.requires_many_years,
        responsibilities=responsibilities,
        hard_constraints=hard_constraints,
        extraction_source="rule_based_structured_extractor"
    )

def extract_jd_structured(text: str) -> StructuredJDInfo:
    return extract_jd_with_rules(text)

def save_extraction(info: StructuredJDInfo, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(info), f, indent=2, ensure_ascii=False)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    jd_text = read_text("data/job_descriptions/sample_jd_1.txt")
    info = extract_jd_structured(jd_text)
    print(asdict(info))
