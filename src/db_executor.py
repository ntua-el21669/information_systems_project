"""
db_executor.py

Διαχειρίζεται τη σύνδεση και εκτέλεση SQL πάνω σε MySQL/MariaDB, και
παρέχει τα εργαλεία που χρειάζεται το evaluation pipeline.

Απαιτεί: pip install pymysql
"""

import time
import pymysql
import pymysql.cursors


def connect(host: str, port: int, user: str, password: str, database: str):
    """Δημιουργεί και επιστρέφει μια σύνδεση σε MySQL ή MariaDB."""
    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        cursorclass=pymysql.cursors.Cursor,
        connect_timeout=10,
    )


MYSQL_CONFIG = {"host": "localhost", "port": 3306, "user": "root", "password": "1234"}
MARIADB_CONFIG = {"host": "localhost", "port": 3307, "user": "root", "password": "1234"}


def check_connection(config: dict, database: str = "geography") -> tuple:
    try:
        conn = connect(database=database, **config)
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchall()
        conn.close()
        return True, None
    except Exception as e:
        return False, str(e)


def run_sql(connection, sql: str, query_timeout_seconds: int = 10) -> dict:
    """
    ΣΗΜΑΝΤΙΚΟ: το "SET SESSION MAX_EXECUTION_TIME=..." είναι MySQL-specific
    εντολή -- το MariaDB δεν την αναγνωρίζει. Αν το αφήναμε μέσα στο ίδιο try/except με το
    πραγματικό query, μια αποτυχία εδώ θα έκανε τα queries να
    αποτυγχάνουν αμέσως όταν τρέχουμε πάνω σε MariaDB.
    """
    start_time = time.time()

    # Best-effort timeout setting
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SET SESSION MAX_EXECUTION_TIME={query_timeout_seconds * 1000}")
    except Exception:
        pass

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

        elapsed = time.time() - start_time
        return {"rows": rows, "columns": columns, "latency_seconds": elapsed, "error": None}

    except Exception as e:
        elapsed = time.time() - start_time
        return {"rows": None, "columns": None, "latency_seconds": elapsed, "error": str(e)}


def get_schema_description(connection, database: str) -> str:
    """
    Παράγει αυτόματα περιγραφή schema (tables + columns) από
    information_schema
    """
    query = """
        SELECT TABLE_NAME, COLUMN_NAME
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """
    with connection.cursor() as cursor:
        cursor.execute(query, (database,))
        rows = cursor.fetchall()

    tables = {}
    for table_name, column_name in rows:
        tables.setdefault(table_name, []).append(column_name)

    lines = [f"Table: {t}({', '.join(cols)})" for t, cols in tables.items()]
    return "\n".join(lines)


def get_table_names(connection, database: str) -> list:
    """Επιστρέφει τη λίστα με τα table names μιας βάσης."""
    query = """
        SELECT DISTINCT TABLE_NAME
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s
    """
    with connection.cursor() as cursor:
        cursor.execute(query, (database,))
        return [row[0] for row in cursor.fetchall()]


def normalize_table_case(sql: str, real_table_names: list) -> str:
    """
    Διορθώνει το case των table names μέσα σε ένα SQL string
    """
    import re
    result = sql
    for table_name in real_table_names:
        pattern = r'\b' + re.escape(table_name) + r'\b'
        result = re.sub(pattern, table_name, result, flags=re.IGNORECASE)
    return result


def _normalize_value(value):
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return value


def compare_execution_results(rows_a, rows_b) -> bool:
    """Συγκρίνει δύο σύνολα αποτελεσμάτων ανεξάρτητα από τη σειρά."""
    if rows_a is None or rows_b is None:
        return False

    def normalize_rows(rows):
        return set(tuple(_normalize_value(v) for v in row) for row in rows)

    return normalize_rows(rows_a) == normalize_rows(rows_b)


def compare_execution_results_lenient(generated_rows, gold_rows) -> bool:
    """
    lenient σύγκριση: True αν κάθε ΜΟΝΑΔΙΚΗ γραμμή του gold
    βρίσκεται σε κάποια γραμμή του generated
    """
    if generated_rows is None or gold_rows is None:
        return False

    gold_unique = {tuple(_normalize_value(v) for v in row) for row in gold_rows}
    gen_unique = {tuple(_normalize_value(v) for v in row) for row in generated_rows}

    if not gold_unique:
        return not gen_unique

    gold_sets = [frozenset(row) for row in gold_unique]
    gen_sets = [frozenset(row) for row in gen_unique]

    remaining = gen_sets.copy()
    for gold_row_set in gold_sets:
        match_index = None
        for i, gen_row_set in enumerate(remaining):
            if gold_row_set.issubset(gen_row_set):
                match_index = i
                break
        if match_index is None:
            return False
        remaining.pop(match_index)

    return True