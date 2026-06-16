from dataclasses import asdict
from structured_jd_extractor import extract_jd_structured

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    path = "data/job_descriptions/sample_jd_4_complex_format.txt"
    jd_text = read_text(path)
    info = extract_jd_structured(jd_text)
    result = asdict(info)
    print(result)
