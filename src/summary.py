"""
summary.py

Διαβάζει όλα τα processed CSV datasets (data/processed/**/*.csv,
συμπεριλαμβανομένων υποφακέλων) και δείχνει μια συγκεντρωτική εικόνα:
    - πόσες γραμμές έχει κάθε dataset
    - κατανομή δυσκολίας (easy/medium/hard) ανά dataset
    - συνολικά νούμερα
"""

import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

OUTPUT_PATH = PROCESSED_DIR / "all_datasets_combined.csv"


def load_all_processed() -> pd.DataFrame:
    """Διαβάζει όλα τα *.csv μέσα στο data/processed/ (και υποφακέλους) και τα ενώνει σε ένα DataFrame."""
    output_path = OUTPUT_PATH.resolve()
    csv_files = [
        f for f in sorted(PROCESSED_DIR.rglob("*.csv"))
        if f.resolve() != output_path
    ]
    if not csv_files:
        raise FileNotFoundError(
            f"No per-dataset .csv files found inside {PROCESSED_DIR} "
            f"(excluding the combined output {OUTPUT_PATH.name}). "
            "Run data_loader.py first for each dataset."
        )

    dfs = []
    print("Input files (the combined output is excluded so re-runs stay idempotent):")
    for csv_file in csv_files:
        part = pd.read_csv(csv_file)
        print(f"  {csv_file.as_posix()}: {len(part)} rows")
        dfs.append(part)
    print()

    combined = pd.concat(dfs, ignore_index=True)

    # αν ξαναμπεί κάποιο aggregate αρχείο στα inputs,
    # το άθροισμα των μερών δεν θα συμφωνεί και θα σκάσει εδώ
    expected = sum(len(part) for part in dfs)
    if len(combined) != expected:
        raise AssertionError(
            f"Combined row count {len(combined)} != sum of parts {expected}."
        )
    return combined


if __name__ == "__main__":
    df = load_all_processed()

    print(f"Total rows (all datasets combined): {len(df)}")
    print()

    print("Rows per dataset:")
    print(df["dataset"].value_counts())
    print()

    print("Difficulty distribution per dataset (counts):")
    crosstab = pd.crosstab(df["dataset"], df["difficulty"])
    print(crosstab)
    print()

    print("Difficulty distribution per dataset (%):")
    crosstab_pct = pd.crosstab(df["dataset"], df["difficulty"], normalize="index") * 100
    print(crosstab_pct.round(1))
    print()

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved combined file: {OUTPUT_PATH} ({len(df)} rows)")