"""
data_loader.py

Μετατρέπει ένα JSON αρχείο από το text2sql-data repo
(π.χ. geography.json, atis.json, restaurants.json) σε μια
απλή, ενιαία μορφή: μία γραμμή ανά (ερώτηση, SQL) ζευγάρι,
με τις μεταβλητές ήδη αντικατεστημένες με πραγματικές τιμές.

Μορφή εισόδου (text2sql-data):
[
  {
    "query-split": "dev",
    "sentences": [
        {"question-split": "dev", "text": "how big is state_name0",
         "variables": {"state_name0": "texas"}},
        ...
    ],
    "sql": [
        "SELECT STATEalias0.AREA FROM STATE AS STATEalias0 "
        "WHERE STATEalias0.STATE_NAME = \"state_name0\" ;"
    ],
    "variables": [
        {"example": "texas", "location": "both",
         "name": "state_name0", "type": "state_name"}
    ]
  },
  ...
]

Έξοδος: pandas DataFrame / CSV με στήλες:
    dataset, query_split, question_split, question, gold_sql
"""

import json
import re
import pandas as pd
from pathlib import Path


def fill_template(template: str, variables: dict) -> str:
    """
    Αντικαθιστά μέσα σε ένα string (SQL ή question) κάθε εμφάνιση
    ενός ονόματος μεταβλητής (π.χ. state_name0) με την πραγματική
    τιμή του (π.χ. texas), βάζοντας quotes γύρω από την τιμή στο SQL
    ΜΟΝΟ αν στο template η μεταβλητή βρισκόταν ήδη μέσα σε quotes.

    Απλή προσέγγιση: αντικαθιστούμε "var_name" (με τα εισαγωγικά
    του template) με "value" (νέα εισαγωγικά + τιμή), και όποιο
    var_name έχει μείνει χωρίς quotes (π.χ. μέσα σε ερώτηση) το
    αντικαθιστούμε σκέτο.
    """
    result = template
    for var_name, var_value in variables.items():
        # 1. Περίπτωση: η μεταβλητή εμφανίζεται μέσα σε διπλά εισαγωγικά
        #    στο SQL, π.χ.  "state_name0"
        result = result.replace(f'"{var_name}"', f'"{var_value}"')
        # 2. Ό,τι έμεινε χωρίς εισαγωγικά (π.χ. σε ερώτηση σε φυσική
        #    γλώσσα, ή αριθμητική μεταβλητή στο SQL) -> απλή αντικατάσταση
        #    λέξης-προς-λέξη (word boundary) ώστε να μην πειράξουμε
        #    π.χ. state_name0 μέσα σε state_name01 (δεν υπάρχει εδώ,
        #    αλλά είναι καλή πρακτική).
        result = re.sub(rf'\b{re.escape(var_name)}\b', str(var_value), result)
    return result


def estimate_difficulty(sql: str) -> str:
    """
    Εκτιμά αυτόματα το επίπεδο δυσκολίας ενός SQL query
    ("easy" / "medium" / "hard") με βάση απλά, μετρήσιμα
    χαρακτηριστικά πολυπλοκότητας:

        - πόσα tables εμπλέκονται (μέσω "AS <alias>")
        - αν υπάρχει nested subquery (δεύτερο SELECT)
        - αν χρησιμοποιείται aggregate function (COUNT/MAX/MIN/AVG/SUM)
        - αν υπάρχει GROUP BY / HAVING / ORDER BY
        - πόσες συνθήκες υπάρχουν στο WHERE (μέσω AND/OR)

    Κάθε χαρακτηριστικό προσθέτει "πόντους" σε ένα σκορ
    πολυπλοκότητας, και το τελικό σκορ μεταφράζεται σε κατηγορία.
    Δεν είναι "τέλειο" (δεν κάνει πραγματικό SQL parsing), αλλά
    δίνει μια λογική, αναπαραγώγιμη εκτίμηση δυσκολίας.
    """
    sql_upper = sql.upper()

    score = 0

    # 1. Αριθμός tables (μετράμε πόσες φορές εμφανίζεται "AS <όνομα>alias")
    num_tables = len(re.findall(r'\bAS\s+\w+', sql_upper))
    if num_tables >= 2:
        score += 1
    if num_tables >= 4:
        score += 1

    # 2. Nested subquery: υπάρχει δεύτερο SELECT μέσα στο query;
    num_selects = len(re.findall(r'\bSELECT\b', sql_upper))
    if num_selects >= 2:
        score += 2  # subqueries μετράνε "βαριά" στην πολυπλοκότητα

    # 3. Aggregate functions
    aggregate_functions = ['COUNT(', 'MAX(', 'MIN(', 'AVG(', 'SUM(']
    if any(fn in sql_upper for fn in aggregate_functions):
        score += 1

    # 4. GROUP BY / HAVING / ORDER BY
    if 'GROUP BY' in sql_upper or 'HAVING' in sql_upper:
        score += 1
    if 'ORDER BY' in sql_upper:
        score += 1

    # 5. Πλήθος συνθηκών στο WHERE (μέσω AND / OR εκτός subquery keywords)
    num_conditions = len(re.findall(r'\bAND\b|\bOR\b', sql_upper))
    if num_conditions >= 2:
        score += 1
    if num_conditions >= 4:
        score += 1

    # Μετατροπή σκορ -> κατηγορία δυσκολίας
    if score <= 1:
        return "easy"
    elif score <= 3:
        return "medium"
    else:
        return "hard"


def load_dataset(json_path: str, dataset_name: str) -> pd.DataFrame:
    """
    Διαβάζει ένα *.json αρχείο μορφής text2sql-data και επιστρέφει
    ένα DataFrame με μία γραμμή ανά πραγματικό (ερώτηση, SQL) ζευγάρι.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    rows = []
    for entry in entries:
        query_split = entry.get("query-split", "")
        # Χρησιμοποιούμε πάντα το ΠΡΩΤΟ SQL variant, όπως προτείνει
        # το README του text2sql-data ("we only use the first query")
        sql_template = entry["sql"][0]

        for sentence in entry["sentences"]:
            question_template = sentence["text"]
            variables = sentence.get("variables", {})

            real_question = fill_template(question_template, variables)
            real_sql = fill_template(sql_template, variables)

            rows.append({
                "dataset": dataset_name,
                "query_split": query_split,
                "question_split": sentence.get("question-split", ""),
                "question": real_question,
                "gold_sql": real_sql,
                "difficulty": estimate_difficulty(real_sql),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Λίστα με όλα τα public datasets που θέλουμε να επεξεργαστούμε.
    # Κάθε entry: (όνομα dataset, path στο .json input, path στο .csv output)
    DATASETS = [
        ("geography", "data/raw/geography/geography.json", "data/processed/geography.csv"),
        ("atis",      "data/raw/atis/atis.json",            "data/processed/atis.csv"),
        ("advising",  "data/raw/advising/advising.json",    "data/processed/advising.csv"),
    ]

    for dataset_name, input_path_str, output_path_str in DATASETS:
        input_path = Path(input_path_str)
        output_path = Path(output_path_str)

        if not input_path.exists():
            print(f"[SKIP] {dataset_name}: input file not found at {input_path}")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = load_dataset(str(input_path), dataset_name=dataset_name)
        df.to_csv(output_path, index=False)

        print(f"[OK] {dataset_name}: loaded {len(df)} (question, SQL) pairs "
              f"from {input_path} -> saved to {output_path}")

    print()
    print("Done processing all datasets.")