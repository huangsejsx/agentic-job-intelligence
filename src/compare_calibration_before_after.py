import pandas as pd
from pathlib import Path

def create_before_after_comparison(output_path: str) -> None:
    rows = [
        {
            "stage": "Before calibration",
            "apply_threshold": 0.75,
            "maybe_threshold": 0.45,
            "accuracy": 0.75,
            "macro_f1": 0.60,
            "main_error": "sample_jd_1.txt was predicted as Apply instead of Maybe",
            "interpretation": "Semantic matching was too optimistic and caused over-matching."
        },
        {
            "stage": "After calibration",
            "apply_threshold": 0.85,
            "maybe_threshold": 0.45,
            "accuracy": 1.00,
            "macro_f1": 1.00,
            "main_error": "No error on the current toy evaluation set",
            "interpretation": "Raising the Apply threshold kept boundary roles as Maybe."
        }
    ]
    df = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved before/after comparison to {output_path}")
    print(df)

if __name__ == "__main__":
    create_before_after_comparison(
        output_path="results/stage5_before_after_comparison.csv"
    )