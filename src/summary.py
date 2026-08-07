"""
summary.py

Διαβάζει όλα τα processed CSV datasets (data/processed/**/*.csv,
συμπεριλαμβανομένων υποφακέλων) και δείχνει μια συγκεντρωτική εικόνα:
    - πόσες γραμμές έχει κάθε dataset
    - κατανομή δυσκολίας (easy/medium/hard) ανά dataset
    - συνολικά νούμερα

Τρέξε το ΑΦΟΥ έχεις ήδη τρέξει τον data_loader.py και για τα 3
datasets (geography, atis, advising), ώστε να υπάρχουν τα
αντίστοιχα .csv αρχεία μέσα στο data/processed/.

Σημείωση: όλα τα print() είναι σκόπιμα στα Αγγλικά, για να
αποφύγουμε UnicodeEncodeError σε Windows terminals (cp1252).
"""

import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def load_all_processed() -> pd.DataFrame:
    """Διαβάζει όλα τα *.csv μέσα στο data/processed/ (και υποφακέλους) και τα ενώνει σε ένα DataFrame."""
    # rglob (αντί για glob) ψάχνει και μέσα σε υποφακέλους,
    # π.χ. data/processed/geography/geography.csv
    csv_files = sorted(PROCESSED_DIR.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No .csv files found inside {PROCESSED_DIR}. "
            "Run data_loader.py first for each dataset."
        )
    dfs = [pd.read_csv(f) for f in csv_files]
    return pd.concat(dfs, ignore_index=True)


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

    # Αποθήκευση του ενοποιημένου dataset (χρήσιμο για το επόμενο βήμα:
    # το evaluation pipeline θα διαβάζει αυτό το ένα αρχείο αντί για 3 ξεχωριστά)
    output_path = Path("data/processed/all_datasets_combined.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved combined file: {output_path}")