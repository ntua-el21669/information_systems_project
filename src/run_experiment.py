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


SAMPLE_SIZE = 300
RANDOM_SEED = 42
LLM_FUNCTION = generate_sql_gpt
LLM_LABEL = "gpt-4o-mini"
DB_CONFIG = MYSQL_CONFIG
DB_LABEL = "mysql"

INPUT_PATH = "data/processed/all_datasets_combined.csv"
OUTPUT_DIR = Path("data/results")

# Τα 3 δημόσια datasets (σε αντίθεση με τα custom_* -- δικά μας queries)
PUBLIC_DATASETS = {"geography", "atis", "advising"}


def stratified_sample(df: pd.DataFrame, total_n: int, random_state: int = 42) -> pd.DataFrame:
    """Στρωματοποιημένο δείγμα διατηρώντας αναλογίες (dataset, difficulty)."""
    fraction = total_n / len(df)
    sampled_parts = []

    for (dataset_name, difficulty), group in df.groupby(["dataset", "difficulty"]):
        n = max(1, round(len(group) * fraction))
        n = min(n, len(group))
        sampled_parts.append(group.sample(n=n, random_state=random_state))

    return pd.concat(sampled_parts, ignore_index=True)


def build_full_sample(df: pd.DataFrame, total_n: int = SAMPLE_SIZE,
                       random_state: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Στρωματοποιημένο δείγμα από τα 3 δημόσια datasets (~total_n ερωτήσεις,
    διατηρώντας αναλογίες dataset x difficulty), ΣΥΝ ΟΛΑ τα custom_* queries
    (31 συνολικά) -- ώστε τα custom queries (ρητή απαίτηση της εκφώνησης)
    να μην "χάνονται" μέσα σε ένα τυχαίο δείγμα λόγω του πολύ μικρού τους
    μεγέθους σε σχέση με τα δημόσια datasets.

    ΣΗΜΑΝΤΙΚΟ: αυτή είναι η ΜΟΝΑΔΙΚΗ πηγή αλήθειας για το ποιες ερωτήσεις
    μπαίνουν στο πείραμα -- τόσο το run_experiment.py (GPT) όσο και το
    export_sample_for_qwen.py (Qwen) καλούν ΑΚΡΙΒΩΣ αυτή τη function, ώστε
    τα δύο μοντέλα να αξιολογούνται πάντα πάνω στις ΙΔΙΕΣ ερωτήσεις --
    προϋπόθεση για το paired McNemar test στο analyze_results.py (το οποίο
    άλλωστε το επαληθεύει ρητά και σκάει με error αν δεν ταιριάζουν).
    """
    public_df = df[df["dataset"].isin(PUBLIC_DATASETS)]
    custom_df = df[~df["dataset"].isin(PUBLIC_DATASETS)]

    public_sample = stratified_sample(public_df, total_n, random_state=random_state)
    return pd.concat([public_sample, custom_df], ignore_index=True)


def print_summary(results_df: pd.DataFrame) -> None:
    # Βαθμολογήσιμα (scoreable) items: το gold SQL έτρεξε χωρίς error.
    # ΣΗΜΕΙΩΣΗ: ΔΕΝ αποκλείουμε πλέον queries όπου gold=0 γραμμές -- οι
    # διορθωμένες compare_execution_results/_lenient() στο db_executor.py
    # τις χειρίζονται ήδη σωστά (trivial_empty_match αν ΚΑΙ generated=0
    # γραμμές, correct=False αν το generated επέστρεψε κάτι διαφορετικό).
    # Ο μόνος πραγματικά "unscoreable" λόγος είναι το ίδιο το gold SQL να
    # μην εκτελείται (gold_execution_error) -- τότε δεν έχουμε καν σημείο
    # αναφοράς για σύγκριση.
    if "gold_execution_error" in results_df.columns:
        scoreable = results_df["gold_execution_error"].isna()
        n_dropped = int((~scoreable).sum())
        results_df = results_df[scoreable]
        print(f"Unscoreable items dropped (gold SQL execution error): {n_dropped}")
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

    # Lenient accuracy: αν υπάρχει η στήλη (νέα runs μετά το few-shot/lenient fix).
    # ΧΩΡΙΣ trivial empty matches, ακριβώς όπως και το strict παραπάνω -- αλλιώς
    # οι δύο μετρικές δεν είναι συγκρίσιμες μεταξύ τους.
    if "correct_lenient" in results_df.columns:
        n_lenient = (results_df["correct_lenient"]
                     & ~results_df["trivial_empty_match"]).sum()
        print(f"Accuracy (LENIENT -- ανεκτικό σε επιπλέον στήλες, "
              f"ΧΩΡΙΣ trivial empty matches): "
              f"{n_lenient}/{total} ({100 * n_lenient / total:.1f}%)")

    print()
    print("Accuracy ανά dataset (χωρίς trivial empty matches):")
    for dataset_name, group in results_df.groupby("dataset"):
        n = len(group)
        correct_real = (group["correct"] & ~group["trivial_empty_match"]).sum()
        line = f"  {dataset_name:20s}: {correct_real}/{n} ({100 * correct_real / n:.1f}%)"
        if "correct_lenient" in results_df.columns:
            lenient_n = (group["correct_lenient"] & ~group["trivial_empty_match"]).sum()
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
    print(f"επιβεβαίωση σύνδεσης στη βάση ({DB_LABEL})...")
    ok, error = check_connection(DB_CONFIG)
    if not ok:
        print("Έλεγξε ότι τα Docker containers τρέχουν: docker ps")
        sys.exit(1)
    print("η βάση είναι προσβάσιμη.")
    print()

    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Total rows available: {len(df)}")

    print(f"Selecting stratified sample of ~{SAMPLE_SIZE} rows from public datasets, "
          f"PLUS all custom_* queries...")
    sample_df = build_full_sample(df, SAMPLE_SIZE, RANDOM_SEED)
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