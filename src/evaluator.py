"""
evaluator.py

Ενώνει το llm_client.py και το db_executor.py σε ένα πλήρες evaluation
βήμα ανά ερώτηση, και batch functions για πολλές ερωτήσεις μαζί.
"""

import time
import pandas as pd

from db_executor import connect, run_sql, get_schema_description, compare_execution_results
from db_executor import compare_execution_results_lenient
from db_executor import get_table_names, normalize_table_case
from db_executor import MYSQL_CONFIG, MARIADB_CONFIG


DATASET_TO_DATABASE = {
    "geography": "geography",
    "atis": "atis",
    "advising": "advising",
    "custom_geography": "geography",
    "custom_atis": "atis",
    "custom_advising": "advising",
}


# Ένα few-shot παράδειγμα ανά πραγματικό schema, ώστε το LLM να μάθει τη
# "γλώσσα" της κάθε βάσης (σωστό case στα table names, στυλ των τιμών,
# και -στο Advising- το μη προφανές navigation μέσω βοηθητικών tables).
# Αυτό προστίθεται στο schema_description ΜΟΝΟ κατά τη ζωντανή κλήση LLM
# (evaluate_single) -- δεν επηρεάζει το evaluate_batch_pregenerated, αφού
# εκεί το SQL έχει ήδη παραχθεί αλλού (π.χ. Qwen/Colab, το οποίο πρέπει
# να πάρει το ΙΔΙΟ augmented schema από το export_sample_for_qwen.py).
FEW_SHOT_EXAMPLES = {
    "geography": {
        "question": "how big is texas",
        "sql": "SELECT area FROM state WHERE state_name = 'texas';",
    },
    "atis": {
        "question": "which airports serve boston",
        "sql": ("SELECT DISTINCT airport_code FROM airport_service "
                "JOIN city ON airport_service.city_code = city.city_code "
                "WHERE city.city_name = 'BOSTON';"),
    },
    "advising": {
        "question": "what courses are related to artificial intelligence",
        "sql": ("SELECT DISTINCT c.NAME FROM COURSE c "
                "JOIN AREA a ON c.COURSE_ID = a.COURSE_ID "
                "WHERE a.AREA LIKE '%artificial intelligence%';"),
    },
}


def build_augmented_schema(database_name: str, schema_description: str) -> str:
    """Προσθέτει ένα few-shot παράδειγμα στο τέλος του schema description."""
    example = FEW_SHOT_EXAMPLES.get(database_name)
    if not example:
        return schema_description
    return (
        f"{schema_description}\n\n"
        f"Example:\n"
        f"Question: {example['question']}\n"
        f"SQL: {example['sql']}"
    )


class SchemaCache:
    def __init__(self):
        self._cache = {}
        self._table_names_cache = {}

    def get(self, connection, database: str) -> str:
        if database not in self._cache:
            self._cache[database] = get_schema_description(connection, database)
        return self._cache[database]

    def get_table_names(self, connection, database: str) -> list:
        if database not in self._table_names_cache:
            self._table_names_cache[database] = get_table_names(connection, database)
        return self._table_names_cache[database]


def score_generated_sql(dataset_label: str, gold_sql: str, generated_sql: str,
                         db_config: dict, connections: dict, schema_cache: SchemaCache) -> dict:
    """Εκτελεί gold + generated SQL, συγκρίνει, επιστρέφει μετρικές εκτέλεσης."""
    database_name = DATASET_TO_DATABASE.get(dataset_label, dataset_label)

    if database_name not in connections:
        connections[database_name] = connect(database=database_name, **db_config)
    connection = connections[database_name]

    result = {
        "execution_error": None,
        "gold_execution_error": None,
        "execution_latency_seconds": None,
        "correct": False,
        "correct_lenient": False,
        "trivial_empty_match": False,
    }

    # NaN-safe έλεγχος "δεν υπάρχει SQL" (το NaN είναι truthy στην Python)
    if generated_sql is None or (isinstance(generated_sql, float) and pd.isna(generated_sql)) \
            or (isinstance(generated_sql, str) and generated_sql.strip() == ""):
        result["execution_error"] = "No SQL generated (LLM call failed)"
        return result

    # Διόρθωση case στα table names (π.χ. "STATE" -> "state") ΠΡΙΝ την
    # εκτέλεση, ώστε το gold SQL (που κληρονομεί το case-convention του
    # πρωτότυπου .json dataset) να μπορεί πράγματι να εκτελεστεί στη δική
    # μας βάση -- βλ. σχόλιο στο normalize_table_case() για λεπτομέρειες.
    real_table_names = schema_cache.get_table_names(connection, database_name)
    gold_sql = normalize_table_case(gold_sql, real_table_names)
    generated_sql = normalize_table_case(generated_sql, real_table_names)

    gold_exec = run_sql(connection, gold_sql)
    if gold_exec["error"] is not None:
        result["gold_execution_error"] = gold_exec["error"]
        return result

    generated_exec = run_sql(connection, generated_sql)
    result["execution_latency_seconds"] = generated_exec["latency_seconds"]

    if generated_exec["error"] is not None:
        result["execution_error"] = generated_exec["error"]
        result["correct"] = False
        return result

    result["correct"] = compare_execution_results(generated_exec["rows"], gold_exec["rows"])
    result["correct_lenient"] = compare_execution_results_lenient(
        generated_exec["rows"], gold_exec["rows"]
    )

    gold_is_empty = len(gold_exec["rows"]) == 0
    generated_is_empty = len(generated_exec["rows"]) == 0
    result["trivial_empty_match"] = bool(
        result["correct"] and gold_is_empty and generated_is_empty
    )

    return result


def evaluate_single(row: dict, generate_sql_fn, db_config: dict, connections: dict,
                     schema_cache: SchemaCache) -> dict:
    """Αξιολογεί μία ερώτηση: LLM -> εκτέλεση -> σύγκριση."""
    dataset_label = row["dataset"]
    database_name = DATASET_TO_DATABASE.get(dataset_label, dataset_label)

    if database_name not in connections:
        connections[database_name] = connect(database=database_name, **db_config)
    connection = connections[database_name]

    schema_description = schema_cache.get(connection, database_name)
    schema_description = build_augmented_schema(database_name, schema_description)

    llm_result = generate_sql_fn(row["question"], schema_description)

    result = {
        "dataset": dataset_label,
        "difficulty": row.get("difficulty"),
        "question": row["question"],
        "gold_sql": row["gold_sql"],
        "generated_sql": llm_result["sql"],
        "llm_model": llm_result["model"],
        "generation_latency_seconds": llm_result["latency_seconds"],
        "generation_error": llm_result["error"],
    }

    score = score_generated_sql(dataset_label, row["gold_sql"], llm_result["sql"],
                                 db_config, connections, schema_cache)
    result.update(score)

    if llm_result["error"] is not None or not llm_result["sql"]:
        result["execution_error"] = "No SQL generated (LLM call failed)"

    return result


def evaluate_batch(df: pd.DataFrame, generate_sql_fn, db_config: dict,
                    limit: int = None, print_progress_every: int = 10) -> pd.DataFrame:
    """Τρέχει evaluate_single πάνω σε πολλές γραμμές, με error isolation."""
    rows_to_process = df if limit is None else df.head(limit)

    connections = {}
    schema_cache = SchemaCache()
    results = []

    start_time = time.time()
    for i, (_, row) in enumerate(rows_to_process.iterrows()):
        try:
            result = evaluate_single(row.to_dict(), generate_sql_fn, db_config,
                                      connections, schema_cache)
        except Exception as e:
            # ΣΗΜΑΝΤΙΚΟ: τυπώνουμε το σφάλμα ΑΜΕΣΩΣ στο terminal, ώστε
            # συστημικά προβλήματα (π.χ. "η βάση δεν είναι προσβάσιμη")
            # να φαίνονται αμέσως, όχι μόνο μετά την ολοκλήρωση/crash.
            print(f"  [!] Unexpected error on row {i}: {e}")
            result = {
                "dataset": row.get("dataset"),
                "difficulty": row.get("difficulty"),
                "question": row.get("question"),
                "gold_sql": row.get("gold_sql"),
                "generated_sql": None,
                "llm_model": None,
                "generation_latency_seconds": None,
                "generation_error": f"Unexpected error: {e}",
                "execution_error": None,
                "gold_execution_error": None,
                "execution_latency_seconds": None,
                "correct": False,
                "correct_lenient": False,
                "trivial_empty_match": False,
            }
        results.append(result)

        if (i + 1) % print_progress_every == 0:
            elapsed = time.time() - start_time
            n_correct = sum(r["correct"] for r in results)
            print(f"  [{i + 1}/{len(rows_to_process)}] "
                  f"accuracy so far: {n_correct}/{i + 1} "
                  f"({100 * n_correct / (i + 1):.1f}%) "
                  f"-- elapsed: {elapsed:.1f}s")

    for conn in connections.values():
        conn.close()

    return pd.DataFrame(results)


def evaluate_batch_pregenerated(df: pd.DataFrame, db_config: dict,
                                 print_progress_every: int = 10) -> pd.DataFrame:
    """Ίδιο με evaluate_batch, αλλά για ΗΔΗ παραγόμενο SQL (π.χ. από Qwen/Colab)."""
    connections = {}
    schema_cache = SchemaCache()
    results = []

    start_time = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        row_dict = row.to_dict()
        try:
            score = score_generated_sql(
                row_dict["dataset"], row_dict["gold_sql"], row_dict.get("generated_sql"),
                db_config, connections, schema_cache,
            )
            result = {
                "dataset": row_dict["dataset"],
                "difficulty": row_dict.get("difficulty"),
                "question": row_dict["question"],
                "gold_sql": row_dict["gold_sql"],
                "generated_sql": row_dict.get("generated_sql"),
                "llm_model": row_dict.get("llm_model") or row_dict.get("model"),
                "generation_latency_seconds": row_dict.get("generation_latency_seconds"),
                "generation_error": row_dict.get("generation_error"),
            }
            result.update(score)
        except Exception as e:
            print(f"  [!] Unexpected error on row {i}: {e}")
            result = {
                "dataset": row_dict.get("dataset"),
                "difficulty": row_dict.get("difficulty"),
                "question": row_dict.get("question"),
                "gold_sql": row_dict.get("gold_sql"),
                "generated_sql": row_dict.get("generated_sql"),
                "llm_model": row_dict.get("llm_model"),
                "generation_latency_seconds": row_dict.get("generation_latency_seconds"),
                "generation_error": f"Unexpected error during scoring: {e}",
                "execution_error": None,
                "gold_execution_error": None,
                "execution_latency_seconds": None,
                "correct": False,
                "correct_lenient": False,
                "trivial_empty_match": False,
            }
        results.append(result)

        if (i + 1) % print_progress_every == 0:
            elapsed = time.time() - start_time
            n_correct = sum(r["correct"] for r in results)
            print(f"  [{i + 1}/{len(df)}] "
                  f"accuracy so far: {n_correct}/{i + 1} "
                  f"({100 * n_correct / (i + 1):.1f}%) "
                  f"-- elapsed: {elapsed:.1f}s")

    for conn in connections.values():
        conn.close()

    return pd.DataFrame(results)