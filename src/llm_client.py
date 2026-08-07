"""
llm_client.py

Ενιαίο interface για να ζητάμε από ένα LLM να μετατρέψει μια ερώτηση
φυσικής γλώσσας σε SQL, δεδομένου του schema μιας βάσης δεδομένων.

Αυτή τη στιγμή περιέχει το GPT-side (μέσω OpenAI API). Το Qwen-side
θα προστεθεί σε ξεχωριστή function (generate_sql_qwen), ώστε το
run_experiment.py να μπορεί να καλεί όποιο LLM θέλει με το ΙΔΙΟ
interface: generate_sql(question, schema_description) -> sql_string.

Απαιτεί:
    pip install openai python-dotenv
    Ένα αρχείο .env στη ρίζα του project με:
        OPENAI_API_KEY=sk-proj-...
"""

import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI

# Φορτώνει το .env αρχείο (πρέπει να υπάρχει OPENAI_API_KEY μέσα του)
load_dotenv()

_client = OpenAI()  # διαβάζει αυτόματα το OPENAI_API_KEY από το environment

DEFAULT_GPT_MODEL = "gpt-4o-mini"  # φθηνό, γρήγορο, αρκετά καλό για text-to-SQL


SYSTEM_PROMPT = """You are a text-to-SQL assistant. Given a database schema \
and a question in natural language, output ONLY the SQL query that answers \
the question. Do not include explanations, comments, or markdown formatting \
(no ```sql fences). Output a single valid SQL statement ending in a semicolon."""


def _build_user_prompt(question: str, schema_description: str) -> str:
    """Συνθέτει το prompt που θα σταλεί στο LLM: schema + ερώτηση."""
    return (
        f"Database schema:\n{schema_description}\n\n"
        f"Question: {question}\n\n"
        f"SQL query:"
    )


def _clean_sql_output(raw_text: str) -> str:
    """
    Καθαρίζει την απάντηση του LLM, αφαιρώντας πιθανά markdown code
    fences (```sql ... ```) ή περιττό whitespace, ώστε να μείνει μόνο
    το καθαρό SQL string.
    """
    text = raw_text.strip()
    # Αφαίρεση ```sql ... ``` ή ``` ... ``` αν υπάρχουν
    text = re.sub(r"^```sql\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    return text.strip()


def generate_sql_gpt(
    question: str,
    schema_description: str,
    model: str = DEFAULT_GPT_MODEL,
    max_retries: int = 3,
) -> dict:
    """
    Καλεί το GPT (μέσω OpenAI API) για να παράξει SQL από μια ερώτηση.

    Επιστρέφει ένα dict με:
        - "sql": το παραγόμενο SQL (string, καθαρισμένο)
        - "latency_seconds": πόσος χρόνος πήρε η κλήση
        - "model": ποιο μοντέλο χρησιμοποιήθηκε
        - "error": None αν όλα πήγαν καλά, αλλιώς μήνυμα σφάλματος
    """
    user_prompt = _build_user_prompt(question, schema_description)

    last_error = None
    for attempt in range(max_retries):
        start_time = time.time()
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,  # θέλουμε ντετερμινιστικές, επαναλήψιμες απαντήσεις
            )
            elapsed = time.time() - start_time

            raw_output = response.choices[0].message.content
            sql = _clean_sql_output(raw_output)

            return {
                "sql": sql,
                "latency_seconds": elapsed,
                "model": model,
                "error": None,
            }

        except Exception as e:
            last_error = str(e)
            elapsed = time.time() - start_time
            # Μικρή αναμονή πριν το retry (exponential backoff)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    # Αν όλες οι προσπάθειες απέτυχαν
    return {
        "sql": None,
        "latency_seconds": elapsed,
        "model": model,
        "error": last_error,
    }


if __name__ == "__main__":
    # Γρήγορο, χειροκίνητο τεστ: ένα απλό schema + μία ερώτηση geography
    test_schema = """
Table: state(state_name, population, area, capital, density)
Table: city(city_name, population, state_name)
Table: river(river_name, length, traverse)
""".strip()

    test_question = "What is the population of Texas?"

    print(f"Question: {test_question}")
    print("Calling GPT...")
    result = generate_sql_gpt(test_question, test_schema)

    print()
    print(f"Model: {result['model']}")
    print(f"Latency: {result['latency_seconds']:.2f}s")
    print(f"Error: {result['error']}")
    print(f"SQL: {result['sql']}")