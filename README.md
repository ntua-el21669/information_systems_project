# Text-to-SQL LLM Comparison Project (LLMSQL3)

Σύγκριση της απόδοσης δύο Large Language Models (**GPT-4o-mini**, **Qwen2.5-Coder-7B-Instruct**) στη μετάφραση φυσικής γλώσσας σε SQL (text-to-SQL), πάνω σε δύο σχεσιακές βάσεις δεδομένων (**MySQL**, **MariaDB**).

## Περιεχόμενα

- [Επισκόπηση](#επισκόπηση)
- [Environment Setup](#environment-setup)
- [Δομή Repository](#δομή-repository)
- [Data Preparation](#data-preparation)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Πώς να τρέξεις τα πάντα από την αρχή](#πώς-να-τρέξεις-τα-πάντα-από-την-αρχή)
- [Αποτελέσματα](#αποτελέσματα)
- [Γνωστοί Περιορισμοί](#γνωστοί-περιορισμοί)

---

## Επισκόπηση

Combo: **LLMSQL3** — GPT, Qwen · MySQL, MariaDB

Το pipeline παίρνει μια ερώτηση σε φυσική γλώσσα, τη στέλνει σε ένα LLM μαζί με το schema μιας βάσης δεδομένων, εκτελεί το παραγόμενο SQL, και συγκρίνει το αποτέλεσμα με ένα γνωστό "σωστό" (gold) SQL — μετρώντας **execution accuracy** και **computational efficiency** (latency).

## Environment Setup

### Βάσεις δεδομένων (Docker)

```bash
docker run --name mysql-db -e MYSQL_ROOT_PASSWORD=1234 -p 3306:3306 -d mysql:8
docker run --name mariadb-db -e MYSQL_ROOT_PASSWORD=1234 -p 3307:3306 -d mariadb:11
```

- MySQL: `localhost:3306`
- MariaDB: `localhost:3307`

### Python dependencies

```bash
pip install -r requirements.txt
```

### OpenAI API key

Δημιούργησε ένα `.env` αρχείο στη ρίζα (δεν committάρεται, βλ. `.gitignore`):

```
OPENAI_API_KEY=sk-proj-...
```

### Qwen (Google Colab)

Το Qwen2.5-Coder-7B-Instruct (4-bit quantized) τρέχει ξεχωριστά σε Google Colab με δωρεάν T4 GPU, αφού δεν υπάρχει τοπική πρόσβαση σε GPU. Βλ. ενότητα [Evaluation Pipeline](#evaluation-pipeline).

---

## Δομή Repository

```
information_systems_project/
├── src/
│   ├── data_loader.py            # JSON -> CSV μετατροπή (3 public datasets) + difficulty labeling
│   ├── custom_queries.py         # 31 δικές μας complex ερωτήσεις (3 schemas)
│   ├── summary.py                # Ενοποίηση όλων των datasets + στατιστικά
│   ├── llm_client.py             # GPT-side: κλήση OpenAI API
│   ├── db_executor.py            # Σύνδεση/εκτέλεση SQL, auto-schema, σύγκριση αποτελεσμάτων
│   ├── evaluator.py               # Ενώνει LLM + DB σε πλήρες evaluation βήμα
│   ├── run_experiment.py         # Στρωματοποιημένο δείγμα + πλήρες GPT run
│   ├── export_sample_for_qwen.py # Εξαγωγή δείγματος (+ schema) για το Colab
│   ├── score_qwen_results.py     # Βαθμολόγηση αποτελεσμάτων Qwen (μετά το Colab)
│   └── rescore_against_db.py     # Επανα-εκτέλεση ήδη-παραγόμενου SQL σε άλλη βάση
├── data/
│   ├── raw/                      # Ωμά datasets (geography, atis, advising: .json + -db.sql)
│   ├── processed/                # Καθαρά CSV (ανά dataset + ενοποιημένο all_datasets_combined.csv)
│   └── results/                  # Τελικά αποτελέσματα αξιολόγησης (4 combos)
├── docs/
│   └── project_notes.docx        # Αναλυτικές σημειώσεις προόδου
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data Preparation

### Πηγή

Τα δημόσια datasets προέρχονται από το [text2sql-data](https://github.com/jkkummerfeld/text2sql-data) repository.

### Datasets

| Dataset | Rows | Χαρακτηριστικά |
|---|---|---|
| Geography | 877 | Γεωγραφία ΗΠΑ, 7 tables, κυρίως απλό |
| ATIS | 5,280 | Πτήσεις/αεροδρόμια, ~25 tables, πολύ σύνθετο |
| Advising | 4,387 | Μαθήματα φοιτητών, 15 tables, ισορροπημένο |
| Custom (3 schemas) | 31 | Δικές μας complex ερωτήσεις, επαληθευμένες χειροκίνητα |

**Σύνολο: 10,575 (ερώτηση, SQL) pairs**, με αυτόματο labeling δυσκολίας (easy/medium/hard) βάσει heuristic πάνω στην πολυπλοκότητα του SQL (πλήθος tables, nested subqueries, aggregate functions, GROUP BY/HAVING κλπ).

### Πώς να αναπαράγεις το data preparation

```bash
python src/data_loader.py        # μετατρέπει geography/atis/advising .json -> .csv
python src/custom_queries.py     # δημιουργεί τα 31 custom queries
python src/summary.py            # ενοποιεί τα πάντα -> all_datasets_combined.csv
```

---

## Evaluation Pipeline

### Αρχιτεκτονική

```
ερώτηση + schema → LLM (GPT/Qwen) → generated SQL
                                          │
                          ┌───────────────┴───────────────┐
                          ▼                                ▼
                  Εκτέλεση στη βάση                Εκτέλεση gold SQL
                          │                                │
                          └───────────────┬────────────────┘
                                           ▼
                         Σύγκριση αποτελεσμάτων (strict + lenient)
```

### Βασικές τεχνικές αποφάσεις

- **Execution accuracy** αντί για string matching (πιο αξιόπιστο — δύο διαφορετικά γραμμένα SQL μπορεί να είναι εξίσου σωστά).
- **Δύο μετρικές ανά ερώτηση**:
  - *Strict*: τα αποτελέσματα πρέπει να ταιριάζουν ακριβώς (ίδιες στήλες/τιμές).
  - *Lenient*: επιτρέπει στο LLM να επιστρέψει επιπλέον στήλες, αρκεί να περιέχουν όλες τις σωστές τιμές.
- **`trivial_empty_match` flag**: αν gold και generated SQL επιστρέφουν *και τα δύο* 0 γραμμές, δεν το μετράμε ως "πραγματικά σωστό" (θα φούσκωνε ψευδώς το accuracy).
- **Αυτόματη παραγωγή schema description** από το ίδιο το `information_schema` της βάσης (όχι χειρόγραφα ανά dataset).
- **Ένα few-shot παράδειγμα ανά schema** μέσα στο prompt, ώστε το LLM να μάθει τις συμβάσεις της κάθε βάσης.
- **Στρωματοποιημένη δειγματοληψία** (306 ερωτήσεις, stratified by dataset × difficulty) για στατιστικά αντιπροσωπευτικό αλλά οικονομικά εφικτό evaluation.

### Πώς να τρέξεις πλήρες evaluation (και τα 4 combos)

```bash
# 1. GPT x MySQL (πραγματική κλήση API)
python src/run_experiment.py

# 2. GPT x MariaDB (reuse του ίδιου SQL, καμία νέα κλήση API)
python src/rescore_against_db.py data/results/results_gpt-4o-mini_mysql.csv mariadb

# 3. Εξαγωγή του ΙΔΙΟΥ δείγματος για Qwen (με schema + few-shot example)
python src/export_sample_for_qwen.py

# 4. Ανέβασε το data/results/sample_for_qwen.csv στο Colab notebook,
#    τρέξε το Qwen (batch mode), κατέβασε το qwen_results.csv

# 5. Qwen x MySQL
python src/score_qwen_results.py

# 6. Qwen x MariaDB (reuse, καμία νέα κλήση Colab)
python src/rescore_against_db.py data/results/results_qwen2.5-coder-7b_mysql.csv mariadb
```

**Σημείωση:** μόνο **2 πραγματικές γεννήσεις SQL** χρειάζονται (1 GPT API pass + 1 Qwen Colab pass) για να καλυφθούν και τα 4 combos, αφού το schema είναι δομικά πανομοιότυπο σε MySQL/MariaDB — το ίδιο SQL απλά ξανατρέχει στη δεύτερη βάση.

---

## Αποτελέσματα

Δείγμα: 306 στρωματοποιημένες ερωτήσεις (ίδιες και για τα 4 combos).

| Metric | GPT×MySQL | GPT×MariaDB | Qwen×MySQL | Qwen×MariaDB |
|---|---|---|---|---|
| Strict accuracy | 13.4% | 13.4% | 10.1% | 10.1% |
| Lenient accuracy | 40.2% | 40.2% | 29.7% | 29.7% |
| Μέσο generation latency | 1.64s | 1.64s | 4.96s | 4.96s |
| Execution errors (syntax) | 42/306 | 42/306 | 98/306 | 99/306 |

### Accuracy ανά dataset (strict / lenient)

| Dataset | GPT | Qwen |
|---|---|---|
| Geography | 60.0% / 60.0% | 64.0% / 64.0% |
| Advising | 11.2% / 60.8% | 5.6% / 44.0% |
| ATIS | 6.7% / 19.5% | 2.7% / 10.7% |

### Βασικά ευρήματα

- **GPT νικά το Qwen** σε accuracy (strict και lenient) και είναι ~3x πιο γρήγορο.
- **Qwen κάνει διπλάσια syntax errors** — αναμενόμενο για ένα μικρότερο, τοπικά τρέχον, quantized μοντέλο.
- **Το RDBMS (MySQL vs MariaDB) δεν επηρεάζει το accuracy** — λογικό, αφού το SQL και τα δεδομένα είναι πανομοιότυπα· επηρεάζει ελαφρώς μόνο το execution latency.
- **Μεγάλο strict→lenient χάσμα στο Advising** (και στα δύο LLMs) — δείχνει ότι τα LLMs συχνά "καταλαβαίνουν" σωστά την ερώτηση αλλά επιστρέφουν επιπλέον/διαφορετικές στήλες απ' ό,τι το gold SQL.
- **ATIS παραμένει δύσκολο ακόμα και στο lenient** — οφείλεται σε γνωστές ιδιαιτερότητες του πρωτότυπου dataset (hardcoded ημερομηνίες 1991, μη-κυριολεκτικές gold απαντήσεις).

---

## Γνωστοί Περιορισμοί

- **Zero-shot / ελάχιστο few-shot**: μόνο 1 παράδειγμα ανά schema· περισσότερα παραδείγματα πιθανότατα θα βελτίωναν σημαντικά το accuracy.
- **Strict metric ευαίσθητο σε επιπλέον στήλες**: ένα σημασιολογικά σωστό SQL μπορεί να "αποτύχει" αν επιστρέφει περισσότερη πληροφορία απ' όσο ζητήθηκε — γι' αυτό αναφέρουμε πάντα strict *και* lenient.
- **`trivial_empty_match`**: δύο ερωτήματα που και τα δύο επιστρέφουν 0 γραμμές (για εντελώς διαφορετικούς/λάθος λόγους) θα μπορούσαν ψευδώς να μετρηθούν ως "ίδια" — το εντοπίζουμε ρητά και το εξαιρούμε από το headline accuracy.
- **Ιδιαιτερότητες πρωτότυπων datasets**: το ATIS/Geography/Advising προέρχονται από ένα ερευνητικό corpus δεκαετιών, με ασυνέπειες case-sensitivity στα table names και ορισμένες gold απαντήσεις που δεν απαντούν κυριολεκτικά στην ερώτηση.
- **Μέγεθος δείγματος**: τα custom datasets (2-15 ερωτήσεις έκαστο) είναι πολύ μικρά για στατιστικά αξιόπιστα ποσοστά μεμονωμένα.

---

## Άδεια χρήσης

Βλ. [LICENSE](LICENSE).