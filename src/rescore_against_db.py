"""
rescore_against_db.py

Παίρνει ένα ΗΔΗ υπάρχον results CSV (π.χ. results_gpt-4o-mini_mysql.csv,
που ήδη περιέχει τη στήλη "generated_sql") και το ξανατρέχει πάνω σε
ΔΙΑΦΟΡΕΤΙΚΗ βάση (π.χ. MariaDB αντί για MySQL).

Γιατί αυτό έχει νόημα: το schema είναι δομικά πανομοιότυπο στο MySQL
και στο MariaDB (φορτώσαμε το ίδιο .sql και στα δύο), άρα το SQL που
παρήγαγε ένα LLM δεν χρειάζεται να ξαναπαραχθεί -- απλά το εκτελούμε
στη δεύτερη βάση και συγκρίνουμε ξανά με το gold. Αυτό μας γλιτώνει
από το να ξανακαλέσουμε το ακριβό/αργό LLM δεύτερη φορά, και επιπλέον
είναι πειραματικά πιο καθαρό: κρατάμε το SQL σταθερό, μεταβάλλουμε
ΜΟΝΟ το RDBMS -- έτσι όποια διαφορά accuracy δούμε οφείλεται καθαρά
σε διαφορές μεταξύ MySQL/MariaDB (π.χ. dialect quirks), όχι σε
διαφορετικό SQL.

Χρήση:
    python src/rescore_against_db.py data/results/results_gpt-4o-mini_mysql.csv mariadb
    python src/rescore_against_db.py qwen_results.csv mariadb
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evaluator import evaluate_batch_pregenerated
from db_executor import MYSQL_CONFIG, MARIADB_CONFIG
from run_experiment import print_summary


DB_CONFIGS = {
    "mysql": MYSQL_CONFIG,
    "mariadb": MARIADB_CONFIG,
}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Χρήση: python src/rescore_against_db.py <input_csv> <mysql|mariadb>")
        sys.exit(1)

    input_path = sys.argv[1]
    db_label = sys.argv[2].lower()

    if db_label not in DB_CONFIGS:
        print(f"Άγνωστο db label: {db_label}. Χρησιμοποίησε 'mysql' ή 'mariadb'.")
        sys.exit(1)

    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"Rows: {len(df)}")

    # Το llm_model label διατηρείται από το input CSV (αν υπάρχει), ώστε το
    # output filename να δείχνει ξεκάθαρα ποιο LLM + ποια βάση συνδυάστηκαν
    llm_label = "unknown_llm"
    if "llm_model" in df.columns and df["llm_model"].notna().any():
        llm_label = str(df["llm_model"].dropna().iloc[0]).replace("/", "-")

    print(f"Rescoring generated SQL (LLM={llm_label}) against {db_label} ...")
    results_df = evaluate_batch_pregenerated(df, DB_CONFIGS[db_label], print_progress_every=20)

    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"results_{llm_label}_{db_label}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    print_summary(results_df)