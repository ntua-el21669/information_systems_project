"""
export_sample_for_qwen.py

Εξάγει το ΙΔΙΟ στρωματοποιημένο δείγμα ερωτήσεων που χρησιμοποιήσαμε ήδη
για το GPT run, μαζί με το schema description (ΠΕΡΙΛΑΜΒΑΝΟΜΕΝΟΥ του ίδιου
few-shot example που βλέπει και το GPT, μέσω build_augmented_schema) --
έτσι το Colab notebook δεν χρειάζεται πρόσβαση στη βάση μας, ΚΑΙ η
σύγκριση GPT vs Qwen παραμένει δίκαιη (ίδιο prompt-context και στα δύο).

Χρήση:
    python src/export_sample_for_qwen.py
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import stratified_sample, SAMPLE_SIZE, RANDOM_SEED, INPUT_PATH
from db_executor import connect, get_schema_description, MYSQL_CONFIG
from evaluator import DATASET_TO_DATABASE, build_augmented_schema


if __name__ == "__main__":
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)

    print(f"Selecting ΙΔΙΟ στρωματοποιημένο δείγμα με το GPT run "
          f"(size~{SAMPLE_SIZE}, seed={RANDOM_SEED}) ...")
    sample_df = stratified_sample(df, SAMPLE_SIZE, random_state=RANDOM_SEED)
    print(f"Sample size: {len(sample_df)}")

    unique_databases = sorted(set(
        DATASET_TO_DATABASE.get(d, d) for d in sample_df["dataset"].unique()
    ))
    print(f"Fetching schema for databases: {unique_databases}")

    schema_by_database = {}
    for database_name in unique_databases:
        conn = connect(database=database_name, **MYSQL_CONFIG)
        raw_schema = get_schema_description(conn, database_name)
        # ΙΔΙΟ few-shot example με αυτό που βλέπει το GPT (evaluate_single),
        # ώστε η σύγκριση GPT vs Qwen να είναι δίκαιη
        schema_by_database[database_name] = build_augmented_schema(database_name, raw_schema)
        conn.close()

    sample_df = sample_df.copy()
    sample_df["schema_description"] = sample_df["dataset"].map(
        lambda d: schema_by_database[DATASET_TO_DATABASE.get(d, d)]
    )

    output_columns = ["dataset", "difficulty", "question", "gold_sql", "schema_description"]
    output_path = Path("data/results/sample_for_qwen.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df[output_columns].to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(sample_df)}")
    print()
    print("Ανέβασε αυτό το αρχείο στο Colab notebook για να τρέξεις το Qwen "
          "πάνω στις ΙΔΙΕΣ ερωτήσεις, με το ΙΔΙΟ few-shot example όπως το GPT.")