"""
run_experiment.py

Στρωματοποιημένη δειγματοληψία + πλήρες evaluation run + συνοπτικά
στατιστικά. Πριν ξεκινήσει το (ενδεχομένως ακριβό/αργό) run, κάνει
ΠΡΟ-ΕΛΕΓΧΟ ότι η βάση είναι πραγματικά προσβάσιμη -- ώστε να μη
σπαταλήσουμε ξανά εκατοντάδες API calls σε ένα run που είναι
καταδικασμένο να αποτύχει λόγω μη διαθέσιμης βάσης.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evaluator import evaluate_batch
from db_executor import MYSQL_CONFIG, MARIADB_CONFIG, check_connection
from llm_client import generate_sql_gpt


SAMPLE_SIZE = 300                      # ΚΛΕΙΔΩΜΕΝΟ -- ίδιο μέγεθος για ΟΛΑ τα 4 combos.
                                        # Μην το αλλάξεις μετά το πρώτο πλήρες run.
RANDOM_SEED = 42
LLM_FUNCTION = generate_sql_gpt
LLM_LABEL = "gpt-4o-mini"
DB_CONFIG = MYSQL_CONFIG
DB_LABEL = "mysql"

INPUT_PATH = "data/processed/all_datasets_combined.csv"
OUTPUT_DIR = Path("data/results")


def stratified_sample(df: pd.DataFrame, total_n: int, random_state: int = 42) -> pd.DataFrame:
    """Στρωματοποιημένο δείγμα διατηρώντας αναλογίες (dataset, difficulty)."""
    fraction = total_n / len(df)
    sampled_parts = []

    for (dataset_name, difficulty), group in df.groupby(["dataset", "difficulty"]):
        n = max(1, round(len(group) * fraction))
        n = min(n, len(group))
        sampled_parts.append(group.sample(n=n, random_state=random_state))

    return pd.concat(sampled_parts, ignore_index=True)


def print_summary(results_df: pd.DataFrame) -> None:
    total = len(results_df)
    n_correct_incl = results_df["correct"].sum()
    n_trivial = results_df["trivial_empty_match"].sum()
    n_correct_excl = n_correct_incl - n_trivial

    print()
    print("=" * 70)
    print("ΣΥΝΟΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ")
    print("=" * 70)
    print(f"Σύνολο ερωτήσεων: {total}")
    print(f"Accuracy (συμπεριλαμβανομένων trivial empty matches): "
          f"{n_correct_incl}/{total} ({100 * n_correct_incl / total:.1f}%)")
    print(f"Accuracy (ΧΩΡΙΣ trivial empty matches, πιο ρεαλιστικό): "
          f"{n_correct_excl}/{total} ({100 * n_correct_excl / total:.1f}%)")
    print(f"Trivial empty matches που αφαιρέθηκαν: {n_trivial}")

    # Lenient accuracy: αν υπάρχει η στήλη (νέα runs μετά το few-shot/lenient fix)
    if "correct_lenient" in results_df.columns:
        n_lenient = results_df["correct_lenient"].sum()
        print(f"Accuracy (LENIENT -- ανεκτικό σε επιπλέον στήλες): "
              f"{n_lenient}/{total} ({100 * n_lenient / total:.1f}%)")

    print()
    print("Accuracy ανά dataset (χωρίς trivial empty matches):")
    for dataset_name, group in results_df.groupby("dataset"):
        n = len(group)
        correct_real = (group["correct"] & ~group["trivial_empty_match"]).sum()
        line = f"  {dataset_name:20s}: {correct_real}/{n} ({100 * correct_real / n:.1f}%)"
        if "correct_lenient" in results_df.columns:
            lenient_n = group["correct_lenient"].sum()
            line += f"   [lenient: {lenient_n}/{n} ({100 * lenient_n / n:.1f}%)]"
        print(line)

    print()
    print("Accuracy ανά επίπεδο δυσκολίας (χωρίς trivial empty matches):")
    for difficulty, group in results_df.groupby("difficulty"):
        n = len(group)
        correct_real = (group["correct"] & ~group["trivial_empty_match"]).sum()
        print(f"  {difficulty:10s}: {correct_real}/{n} ({100 * correct_real / n:.1f}%)")

    print()
    avg_gen_latency = results_df["generation_latency_seconds"].mean()
    avg_exec_latency = results_df["execution_latency_seconds"].mean()
    print(f"Μέσος χρόνος παραγωγής SQL (LLM): {avg_gen_latency:.2f}s")
    print(f"Μέσος χρόνος εκτέλεσης SQL (DB):  {avg_exec_latency:.3f}s")

    n_generation_errors = results_df["generation_error"].notna().sum()
    n_execution_errors = results_df["execution_error"].notna().sum()
    print()
    print(f"Αποτυχίες παραγωγής SQL (π.χ. API errors): {n_generation_errors}/{total}")
    print(f"Αποτυχίες εκτέλεσης SQL (π.χ. syntax errors του LLM): {n_execution_errors}/{total}")


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # PRE-FLIGHT CHECK: επιβεβαίωσε ότι η βάση είναι προσβάσιμη ΠΡΙΝ
    # ξεκινήσουμε εκατοντάδες (ενδεχομένως πληρωμένες) κλήσεις LLM.
    # Αυτό αποτρέπει ακριβώς το πρόβλημα που είχαμε: 306 κλήσεις GPT
    # "στο κενό" επειδή το MySQL container δεν έτρεχε.
    # -----------------------------------------------------------------
    print(f"Pre-flight check: επιβεβαίωση σύνδεσης στη βάση ({DB_LABEL})...")
    ok, error = check_connection(DB_CONFIG)
    if not ok:
        print(f"[ΣΦΑΛΜΑ] Δεν μπορώ να συνδεθώ στη βάση: {error}")
        print("Έλεγξε ότι τα Docker containers τρέχουν: docker ps")
        print("Αν όχι: docker start mysql-db mariadb-db")
        sys.exit(1)
    print("Pre-flight check: OK, η βάση είναι προσβάσιμη.")
    print()

    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Total rows available: {len(df)}")

    print(f"Selecting stratified sample of ~{SAMPLE_SIZE} rows "
          f"(stratified by dataset x difficulty)...")
    sample_df = stratified_sample(df, SAMPLE_SIZE, random_state=RANDOM_SEED)
    print(f"Actual sample size: {len(sample_df)}")
    print()
    print("Sample composition (rows per dataset):")
    print(sample_df["dataset"].value_counts())
    print()

    print(f"Running evaluation: LLM={LLM_LABEL}, DB={DB_LABEL} ...")
    print(f"(This will make {len(sample_df)} real API calls -- monitor cost if using GPT)")
    print()

    results_df = evaluate_batch(sample_df, LLM_FUNCTION, DB_CONFIG, print_progress_every=20)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"results_{LLM_LABEL}_{DB_LABEL}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved full results to: {output_path}")

    print_summary(results_df)