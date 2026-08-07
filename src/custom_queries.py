"""
custom_queries.py

Οι δικές μας, χειρόγραφες complex ερωτήσεις πάνω στο Geography
schema (7 tables: state, city, river, mountain, lake, border_info,
highlow). Φτιάχτηκαν σκόπιμα πιο σύνθετες από το αυθεντικό
geography.json dataset -- πολλαπλά JOINs, nested subqueries με
2+ επίπεδα, GROUP BY + HAVING, συγκρίσεις μεταξύ ομάδων.

Στόχος: κάλυψη της απαίτησης της εκφώνησης για "δημιουργία δικών
σας πιο σύνθετων SQL ερωτημάτων", πέρα από τα έτοιμα public datasets.

Έξοδος: CSV στην ΙΔΙΑ μορφή με τα υπόλοιπα processed datasets
(dataset, query_split, question_split, question, gold_sql, difficulty)
ώστε να ενσωματώνεται απευθείας στο ίδιο evaluation pipeline.

ΣΗΜΑΝΤΙΚΟ: Πριν εμπιστευτείς αυτά τα queries ως "gold" σωστά,
έτρεξέ τα ΧΕΙΡΟΚΙΝΗΤΑ στο DBeaver πάνω στο geography database
(και MySQL και MariaDB) για να επιβεβαιώσεις ότι επιστρέφουν
λογικά αποτελέσματα.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import estimate_difficulty  # noqa: E402


CUSTOM_QUERIES = [
    {
        "question": "Which states border a state whose capital population "
                    "is above the average city population across all states?",
        "gold_sql": """
            SELECT DISTINCT b.STATE_NAME
            FROM border_info b
            JOIN state s ON b.BORDER = s.STATE_NAME
            JOIN city c ON c.CITY_NAME = s.CAPITAL AND c.STATE_NAME = s.STATE_NAME
            WHERE c.POPULATION > (
                SELECT AVG(POPULATION) FROM city
            ) ;
        """.strip(),
    },
    {
        "question": "For each state, list the number of cities with population "
                    "over 100000, but only show states with more than 2 such cities.",
        "gold_sql": """
            SELECT STATE_NAME, COUNT(*) AS big_city_count
            FROM city
            WHERE POPULATION > 100000
            GROUP BY STATE_NAME
            HAVING COUNT(*) > 2 ;
        """.strip(),
    },
    {
        "question": "Which rivers flow through more states than the river "
                    "with the most states, minus one?",
        "gold_sql": """
            SELECT RIVER_NAME
            FROM river
            GROUP BY RIVER_NAME
            HAVING COUNT(DISTINCT TRAVERSE) >= (
                SELECT MAX(state_count) - 1
                FROM (
                    SELECT COUNT(DISTINCT TRAVERSE) AS state_count
                    FROM river
                    GROUP BY RIVER_NAME
                ) AS counts
            ) ;
        """.strip(),
    },
    {
        "question": "List states whose area is larger than the average area "
                    "of states they border.",
        "gold_sql": """
            SELECT s.STATE_NAME
            FROM state s
            WHERE s.AREA > (
                SELECT AVG(s2.AREA)
                FROM border_info b
                JOIN state s2 ON b.BORDER = s2.STATE_NAME
                WHERE b.STATE_NAME = s.STATE_NAME
            ) ;
        """.strip(),
    },
    {
        "question": "Find the state with the highest ratio of population to area "
                    "among states that have at least one mountain.",
        "gold_sql": """
            SELECT s.STATE_NAME, (s.POPULATION / s.AREA) AS density
            FROM state s
            WHERE s.STATE_NAME IN (
                SELECT DISTINCT STATE_NAME FROM mountain
            )
            ORDER BY density DESC
            LIMIT 1 ;
        """.strip(),
    },
    {
        "question": "Which states have both a lake and a mountain, and border "
                    "at least 3 other states?",
        "gold_sql": """
            SELECT DISTINCT s.STATE_NAME
            FROM state s
            WHERE s.STATE_NAME IN (SELECT STATE_NAME FROM lake)
              AND s.STATE_NAME IN (SELECT STATE_NAME FROM mountain)
              AND (
                  SELECT COUNT(*) FROM border_info b
                  WHERE b.STATE_NAME = s.STATE_NAME
              ) >= 3 ;
        """.strip(),
    },
    {
        "question": "For every state, compute the difference between its highest "
                    "and lowest point elevation, and list the top 5 states with "
                    "the largest difference.",
        "gold_sql": """
            SELECT STATE_NAME, (HIGHEST_ELEVATION - LOWEST_ELEVATION) AS elevation_range
            FROM highlow
            ORDER BY elevation_range DESC
            LIMIT 5 ;
        """.strip(),
    },
    {
        "question": "Which cities are the most populous city in their state, "
                    "and also have a population greater than the state's capital?",
        "gold_sql": """
            SELECT c.CITY_NAME, c.STATE_NAME
            FROM city c
            WHERE c.POPULATION = (
                SELECT MAX(c2.POPULATION)
                FROM city c2
                WHERE c2.STATE_NAME = c.STATE_NAME
            )
            AND c.POPULATION > (
                SELECT cap.POPULATION
                FROM city cap
                JOIN state s ON cap.CITY_NAME = s.CAPITAL AND cap.STATE_NAME = s.STATE_NAME
                WHERE s.STATE_NAME = c.STATE_NAME
            ) ;
        """.strip(),
    },
    {
        "question": "List the top 3 states by number of rivers, along with the "
                    "count, excluding states with zero rivers.",
        "gold_sql": """
            SELECT TRAVERSE AS STATE_NAME, COUNT(*) AS river_count
            FROM river
            GROUP BY TRAVERSE
            ORDER BY river_count DESC
            LIMIT 3 ;
        """.strip(),
    },
    {
        "question": "Which pairs of neighboring states both have a population "
                    "density (population/area) above 100?",
        "gold_sql": """
            SELECT b.STATE_NAME, b.BORDER
            FROM border_info b
            JOIN state s1 ON b.STATE_NAME = s1.STATE_NAME
            JOIN state s2 ON b.BORDER = s2.STATE_NAME
            WHERE (s1.POPULATION / s1.AREA) > 100
              AND (s2.POPULATION / s2.AREA) > 100 ;
        """.strip(),
    },
    {
        "question": "Find states where the tallest mountain is taller than the "
                    "tallest mountain in every state they border.",
        "gold_sql": """
            SELECT DISTINCT m.STATE_NAME
            FROM mountain m
            WHERE m.MOUNTAIN_ALTITUDE > ALL (
                SELECT m2.MOUNTAIN_ALTITUDE
                FROM border_info b
                JOIN mountain m2 ON b.BORDER = m2.STATE_NAME
                WHERE b.STATE_NAME = m.STATE_NAME
            ) ;
        """.strip(),
    },
    {
        "question": "What is the average population of cities in states that "
                    "do not have any lakes?",
        "gold_sql": """
            SELECT AVG(c.POPULATION)
            FROM city c
            WHERE c.STATE_NAME NOT IN (
                SELECT DISTINCT STATE_NAME FROM lake
            ) ;
        """.strip(),
    },
    {
        "question": "Which state has the most cities with population under "
                    "50000, and how many such cities does it have?",
        "gold_sql": """
            SELECT STATE_NAME, COUNT(*) AS small_city_count
            FROM city
            WHERE POPULATION < 50000
            GROUP BY STATE_NAME
            ORDER BY small_city_count DESC
            LIMIT 1 ;
        """.strip(),
    },
    {
        "question": "List states where the capital is NOT the most populous "
                    "city in that state.",
        "gold_sql": """
            SELECT s.STATE_NAME
            FROM state s
            WHERE s.CAPITAL != (
                SELECT c.CITY_NAME
                FROM city c
                WHERE c.STATE_NAME = s.STATE_NAME
                ORDER BY c.POPULATION DESC
                LIMIT 1
            ) ;
        """.strip(),
    },
    {
        "question": "Find rivers that traverse a state bordering the state "
                    "with the largest area (among states that have land borders).",
        "gold_sql": """
            SELECT DISTINCT r.RIVER_NAME
            FROM river r
            WHERE r.TRAVERSE IN (
                SELECT b.STATE_NAME
                FROM border_info b
                WHERE b.BORDER = (
                    SELECT STATE_NAME FROM state
                    WHERE STATE_NAME IN (SELECT DISTINCT STATE_NAME FROM border_info)
                    ORDER BY AREA DESC
                    LIMIT 1
                )
            ) ;
        """.strip(),
    },
]


# ---------------------------------------------------------------------------
# ATIS custom queries
#
# ΣΗΜΑΝΤΙΚΟ: Στο ATIS schema, ΤΟΣΟ τα table names ΟΣΟ ΚΑΙ τα column names
# είναι πεζά (π.χ. flight, from_airport) -- διαφορετικά από το Geography,
# όπου μόνο τα tables ήταν πεζά αλλά τα columns κεφαλαία. Επιβεβαιώθηκε
# απευθείας μέσα από το atis-db.sql (CREATE TABLE statements).
#
# Επίσης: string values μέσα στα δεδομένα (π.χ. city names) είναι σε
# ΚΕΦΑΛΑΙΑ (π.χ. 'ATLANTA', 'BOSTON'), σε αντίθεση με το Geography όπου
# ήταν πεζά (π.χ. 'texas').
# ---------------------------------------------------------------------------

ATIS_CUSTOM_QUERIES = [
    {
        "question": "Which airlines operate more flights than the average "
                    "number of flights per airline?",
        "gold_sql": """
            SELECT airline_code, COUNT(*) AS flight_count
            FROM flight
            GROUP BY airline_code
            HAVING COUNT(*) > (
                SELECT AVG(cnt) FROM (
                    SELECT COUNT(*) AS cnt
                    FROM flight
                    GROUP BY airline_code
                ) AS airline_counts
            ) ;
        """.strip(),
    },
    {
        "question": "Which cities are served by more than one airport?",
        "gold_sql": """
            SELECT c.city_name, COUNT(DISTINCT s.airport_code) AS airport_count
            FROM city c
            JOIN airport_service s ON c.city_code = s.city_code
            GROUP BY c.city_name
            HAVING COUNT(DISTINCT s.airport_code) > 1 ;
        """.strip(),
    },
    {
        "question": "For each route (from airport to airport) that has more "
                    "than one airline offering a fare, show the cheapest "
                    "one-direction cost.",
        "gold_sql": """
            SELECT from_airport, to_airport, MIN(one_direction_cost) AS cheapest_fare
            FROM fare
            GROUP BY from_airport, to_airport
            HAVING COUNT(DISTINCT fare_airline) > 1 ;
        """.strip(),
    },
    {
        "question": "Find flights that have more stops than the average "
                    "number of stops among flights operated by the same airline.",
        "gold_sql": """
            SELECT f.flight_id, f.airline_code, f.stops
            FROM flight f
            WHERE f.stops > (
                SELECT AVG(f2.stops)
                FROM flight f2
                WHERE f2.airline_code = f.airline_code
            ) ;
        """.strip(),
    },
    {
        "question": "Which airlines have an average one-direction fare cost "
                    "higher than the overall average across all airlines?",
        "gold_sql": """
            SELECT fare_airline, AVG(one_direction_cost) AS avg_cost
            FROM fare
            GROUP BY fare_airline
            HAVING AVG(one_direction_cost) > (
                SELECT AVG(one_direction_cost) FROM fare
            ) ;
        """.strip(),
    },
    {
        "question": "Which states have airports served by flights from more "
                    "than 3 different airlines?",
        "gold_sql": """
            SELECT st.state_name, COUNT(DISTINCT fl.airline_code) AS airline_count
            FROM state st
            JOIN airport ap ON st.state_code = ap.state_code
            JOIN flight fl ON fl.from_airport = ap.airport_code
            GROUP BY st.state_name
            HAVING COUNT(DISTINCT fl.airline_code) > 3 ;
        """.strip(),
    },
    {
        "question": "Find the flight(s) with the maximum number of stops, "
                    "along with the airline name operating them.",
        "gold_sql": """
            SELECT f.flight_id, al.airline_name, f.stops
            FROM flight f
            JOIN airline al ON f.airline_code = al.airline_code
            WHERE f.stops = (
                SELECT MAX(stops) FROM flight
            ) ;
        """.strip(),
    },
    {
        "question": "List fare basis codes whose average one-direction cost "
                    "is above the overall average one-direction cost.",
        "gold_sql": """
            SELECT fare_basis_code, AVG(one_direction_cost) AS avg_cost
            FROM fare
            GROUP BY fare_basis_code
            HAVING AVG(one_direction_cost) > (
                SELECT AVG(one_direction_cost) FROM fare
            ) ;
        """.strip(),
    },
]


# ---------------------------------------------------------------------------
# Advising custom queries
#
# ΣΗΜΑΝΤΙΚΟ: Στο Advising schema, τα TABLE names είναι ΚΕΦΑΛΑΙΑ
# (επιβεβαιώθηκε με SHOW TABLES: COURSE, STUDENT, PROGRAM, κλπ) --
# διαφορετικά από το ATIS (όλα πεζά) και το Geography (μόνο tables πεζά).
#
# Τα COLUMN names δεν χρειάζεται να ταιριάζουν σε case -- η MySQL δεν
# κάνει ποτέ διάκριση πεζών/κεφαλαίων σε column names, ανεξαρτήτως OS.
# Μόνο τα table names είναι ευαίσθητα σε case (στο Linux, ανάλογα με τη
# server config). Γι' αυτό εδώ γράφουμε τα columns όπως εμφανίζονται στο
# advising-db.sql (μερικά tables π.χ. COURSE/INSTRUCTOR έχουν κεφαλαία
# columns, άλλα π.χ. STUDENT/PROGRAM έχουν πεζά -- και τα δύο δουλεύουν).
# ---------------------------------------------------------------------------

ADVISING_CUSTOM_QUERIES = [
    {
        "question": "Which instructors have taught more course offerings "
                    "than the average number of offerings per instructor?",
        "gold_sql": """
            SELECT oi.INSTRUCTOR_ID, COUNT(*) AS offering_count
            FROM OFFERING_INSTRUCTOR oi
            GROUP BY oi.INSTRUCTOR_ID
            HAVING COUNT(*) > (
                SELECT AVG(cnt) FROM (
                    SELECT COUNT(*) AS cnt
                    FROM OFFERING_INSTRUCTOR
                    GROUP BY INSTRUCTOR_ID
                ) AS counts
            ) ;
        """.strip(),
    },
    {
        "question": "List courses that have more than 2 prerequisite courses.",
        "gold_sql": """
            SELECT c.COURSE_ID, c.NAME, COUNT(*) AS prereq_count
            FROM COURSE_PREREQUISITE cp
            JOIN COURSE c ON cp.course_id = c.COURSE_ID
            GROUP BY c.COURSE_ID, c.NAME
            HAVING COUNT(*) > 2 ;
        """.strip(),
    },
    {
        "question": "Which programs have an average course workload higher "
                    "than the overall average workload across all programs?",
        "gold_sql": """
            SELECT program_id, AVG(workload) AS avg_workload
            FROM PROGRAM_COURSE
            GROUP BY program_id
            HAVING AVG(workload) > (
                SELECT AVG(workload) FROM PROGRAM_COURSE
            ) ;
        """.strip(),
    },
    {
        "question": "Find students whose total GPA is higher than the "
                    "average GPA of students in the same program.",
        "gold_sql": """
            SELECT s.student_id, s.firstname, s.lastname, s.total_gpa
            FROM STUDENT s
            WHERE s.total_gpa > (
                SELECT AVG(s2.total_gpa)
                FROM STUDENT s2
                WHERE s2.program_id = s.program_id
            ) ;
        """.strip(),
    },
    {
        "question": "Which courses have a clarity score above the average "
                    "clarity score for their department?",
        "gold_sql": """
            SELECT c.COURSE_ID, c.NAME, c.DEPARTMENT, c.CLARITY_SCORE
            FROM COURSE c
            WHERE c.CLARITY_SCORE > (
                SELECT AVG(c2.CLARITY_SCORE)
                FROM COURSE c2
                WHERE c2.DEPARTMENT = c.DEPARTMENT
            ) ;
        """.strip(),
    },
    {
        "question": "Find courses that are required as a prerequisite for "
                    "more than 3 other courses.",
        "gold_sql": """
            SELECT c.COURSE_ID, c.NAME, COUNT(*) AS required_for_count
            FROM COURSE_PREREQUISITE cp
            JOIN COURSE c ON cp.pre_course_id = c.COURSE_ID
            GROUP BY c.COURSE_ID, c.NAME
            HAVING COUNT(*) > 3 ;
        """.strip(),
    },
    {
        "question": "Which semesters had more course offerings than the "
                    "average number of offerings per semester?",
        "gold_sql": """
            SELECT SEMESTER, COUNT(*) AS offering_count
            FROM COURSE_OFFERING
            GROUP BY SEMESTER
            HAVING COUNT(*) > (
                SELECT AVG(cnt) FROM (
                    SELECT COUNT(*) AS cnt
                    FROM COURSE_OFFERING
                    GROUP BY SEMESTER
                ) AS counts
            ) ;
        """.strip(),
    },
    {
        "question": "Find courses with more enrolled students than the "
                    "average enrollment across all courses in the same "
                    "department.",
        "gold_sql": """
            SELECT c.COURSE_ID, c.NAME, c.DEPARTMENT, c.NUM_ENROLLED
            FROM COURSE c
            WHERE c.NUM_ENROLLED > (
                SELECT AVG(c2.NUM_ENROLLED)
                FROM COURSE c2
                WHERE c2.DEPARTMENT = c.DEPARTMENT
            ) ;
        """.strip(),
    },
]


def build_custom_dataset() -> pd.DataFrame:
    rows = []

    # Geography custom queries -> dataset label "custom_geography"
    for item in CUSTOM_QUERIES:
        clean_sql = " ".join(item["gold_sql"].split())
        rows.append({
            "dataset": "custom_geography",
            "query_split": "test",
            "question_split": "test",
            "question": item["question"],
            "gold_sql": clean_sql,
            "difficulty": estimate_difficulty(clean_sql),
        })

    # ATIS custom queries -> dataset label "custom_atis"
    for item in ATIS_CUSTOM_QUERIES:
        clean_sql = " ".join(item["gold_sql"].split())
        rows.append({
            "dataset": "custom_atis",
            "query_split": "test",
            "question_split": "test",
            "question": item["question"],
            "gold_sql": clean_sql,
            "difficulty": estimate_difficulty(clean_sql),
        })

    # Advising custom queries -> dataset label "custom_advising"
    for item in ADVISING_CUSTOM_QUERIES:
        clean_sql = " ".join(item["gold_sql"].split())
        rows.append({
            "dataset": "custom_advising",
            "query_split": "test",
            "question_split": "test",
            "question": item["question"],
            "gold_sql": clean_sql,
            "difficulty": estimate_difficulty(clean_sql),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_custom_dataset()

    output_dir = Path("data/processed/custom")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "custom.csv"
    df.to_csv(output_path, index=False)

    print(f"Created {len(df)} custom complex queries")
    print()
    print("Difficulty distribution:")
    print(df["difficulty"].value_counts())
    print()
    print(f"Saved to: {output_path}")
