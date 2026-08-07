"""
score_qwen_results.py

Παίρνει το qwen_results.csv (κατέβηκε από το Colab notebook, ήδη
περιέχει το generated_sql για κάθε ερώτηση), εκτελεί το SQL τοπικά
στη δική μας MySQL/MariaDB, συγκρίνει με το gold SQL, και τυπώνει
ΤΗΝ ΙΔΙΑ αναφορά όπως το run_experiment.py (GPT) -- ώστε τα δύο
αποτελέσματα να είναι απευθείας συγκρίσιμα.

Χρήση:
    python src/score_qwen_results.py
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evaluator import evaluate_batch_pregenerated
from db_executor import MYSQL_CONFIG
from run_experiment import print_summary


INPUT_PATH = "data/results/qwen_results_raw.csv"
DB_CONFIG = MYSQL_CONFIG
DB_LABEL = "mysql"
LLM_LABEL = "qwen2.5-coder-7b"


if __name__ == "__main__":
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Rows: {len(df)}")
    print()

    print(f"Scoring against {DB_LABEL} ...")
    results_df = evaluate_batch_pregenerated(df, DB_CONFIG, print_progress_every=10)

    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"results_{LLM_LABEL}_{DB_LABEL}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved full results to: {output_path}")

    print_summary(results_df)