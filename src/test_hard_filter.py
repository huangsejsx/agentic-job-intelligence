from dataclasses import asdict
from parse_jd import parse_jd

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    path = "data/job_descriptions/sample_jd_2_hard_filter.txt"
    jd_text = read_text(path)
    info = parse_jd(jd_text)
    print(asdict(info))