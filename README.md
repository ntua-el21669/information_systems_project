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
- [Στατιστική Σημαντικότητα](#στατιστική-σημαντικότητα)
- [Γνωστοί Περιορισμοί](#γνωστοί-περιορισμοί)
- [Κατάσταση Παραδοτέων](#κατάσταση-παραδοτέων)

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
│   ├── data_loader.py             # JSON -> CSV μετατροπή (3 public datasets) + difficulty labeling
│   ├── custom_queries.py          # 31 δικές μας complex ερωτήσεις (3 schemas)
│   ├── summary.py                 # Ενοποίηση όλων των datasets + στατιστικά
│   ├── llm_client.py              # GPT-side: κλήση OpenAI API
│   ├── db_executor.py             # Σύνδεση/εκτέλεση SQL, auto-schema, σύγκριση αποτελεσμάτων
│   ├── evaluator.py                # Ενώνει LLM + DB σε πλήρες evaluation βήμα
│   ├── run_experiment.py          # Στρωματοποιημένο δείγμα + πλήρες GPT run
│   ├── export_sample_for_qwen.py  # Εξαγωγή δείγματος (+ schema) για το Colab
│   ├── score_qwen_results.py      # Βαθμολόγηση αποτελεσμάτων Qwen (μετά το Colab)
│   ├── rescore_against_db.py      # Επανα-εκτέλεση ήδη-παραγόμενου SQL σε άλλη βάση
│   └── analyze_results.py         # Στατιστική ανάλυση (Wilson CI, McNemar test) + γραφήματα
├── data/
│   ├── raw/                       # Ωμά datasets (geography, atis, advising: .json + -db.sql)
│   ├── processed/                 # Καθαρά CSV (ανά dataset + ενοποιημένο all_datasets_combined.csv)
│   └── results/
│       ├── results_*.csv          # Τελικά αποτελέσματα αξιολόγησης (4 combos)
│       ├── sample_for_qwen.csv    # Το στρωματοποιημένο δείγμα (306 ερωτήσεις) με schema
│       ├── qwen_results_raw.csv   # Ωμή έξοδος Qwen από το Colab, πριν το scoring
│       └── analysis/              # Στατιστική ανάλυση + γραφήματα
│           ├── accuracy_overall.svg
│           ├── accuracy_by_dataset.svg
│           ├── accuracy_by_difficulty.svg
│           ├── metric_summary.csv
│           ├── paired_mcnemar_test.csv
│           └── statistical_summary.md
├── docs/
│   └── project_notes.docx         # Αναλυτικές σημειώσεις προόδου (χρονολογικά, με κάθε πρόβλημα/λύση)
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
                                           │
                                           ▼
                    analyze_results.py: Wilson CI + McNemar test + γραφήματα
```

### Βασικές τεχνικές αποφάσεις

- **Execution accuracy** αντί για string matching (πιο αξιόπιστο — δύο διαφορετικά γραμμένα SQL μπορεί να είναι εξίσου σωστά).
- **Δύο μετρικές ανά ερώτηση**:
  - *Strict*: τα αποτελέσματα πρέπει να ταιριάζουν ακριβώς (ίδιες στήλες/τιμές).
  - *Lenient*: επιτρέπει στο LLM να επιστρέψει επιπλέον στήλες, αρκεί να περιέχουν όλες τις σωστές τιμές.
- **`trivial_empty_match` flag**: αν gold και generated SQL επιστρέφουν *και τα δύο* 0 γραμμές, δεν το μετράμε ως "πραγματικά σωστό" (θα φούσκωνε ψευδώς το accuracy) — εξαιρείται ρητά από τη στατιστική ανάλυση.
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
#    τρέξε το Qwen (batch mode), κατέβασε το qwen_results_raw.csv

# 5. Qwen x MySQL
python src/score_qwen_results.py

# 6. Qwen x MariaDB (reuse, καμία νέα κλήση Colab)
python src/rescore_against_db.py data/results/results_qwen2.5-coder-7b_mysql.csv mariadb

# 7. Στατιστική ανάλυση + γραφήματα (και τα 4 combos)
python src/analyze_results.py
```

**Σημείωση:** μόνο **2 πραγματικές γεννήσεις SQL** χρειάζονται (1 GPT API pass + 1 Qwen Colab pass) για να καλυφθούν και τα 4 combos, αφού το schema είναι δομικά πανομοιότυπο σε MySQL/MariaDB — το ίδιο SQL απλά ξανατρέχει στη δεύτερη βάση.

---

## Αποτελέσματα

Δείγμα: 306 στρωματοποιημένες ερωτήσεις (ίδιες και για τα 4 combos). Πλήρη στατιστικά στο [`data/results/analysis/statistical_summary.md`](data/results/analysis/statistical_summary.md).

### Overall execution accuracy (95% Wilson confidence intervals)

| Μοντέλο | Μετρική | Αποτέλεσμα | 95% CI |
|---|---|---:|---:|
| GPT-4o-mini | Strict | 41/306 (13.4%) | 10.0%–17.7% |
| GPT-4o-mini | Lenient | 123/306 (40.2%) | 34.9%–45.8% |
| Qwen2.5-Coder-7B-Instruct | Strict | 31/306 (10.1%) | 7.2%–14.0% |
| Qwen2.5-Coder-7B-Instruct | Lenient | 91/306 (29.7%) | 24.9%–35.1% |

### Λοιπές μετρικές

| Metric | GPT×MySQL | GPT×MariaDB | Qwen×MySQL | Qwen×MariaDB |
|---|---|---|---|---|
| Μέσο generation latency | 1.64s | 1.64s | 4.96s | 4.96s |
| Execution errors (syntax) | 42/306 | 42/306 | 98/306 | 99/306 |

### Accuracy ανά dataset και δυσκολία

Βλ. γραφήματα: [`accuracy_by_dataset.svg`](data/results/analysis/accuracy_by_dataset.svg), [`accuracy_by_difficulty.svg`](data/results/analysis/accuracy_by_difficulty.svg).

### Βασικά ευρήματα

- **GPT νικά το Qwen** σε accuracy (strict και lenient) και είναι ~3x πιο γρήγορο.
- **Qwen κάνει διπλάσια syntax errors** — αναμενόμενο για ένα μικρότερο, τοπικά τρέχον, quantized μοντέλο.
- **Το RDBMS (MySQL vs MariaDB) δεν επηρεάζει το accuracy** — λογικό, αφού το SQL και τα δεδομένα είναι πανομοιότυπα· επηρεάζει ελαφρώς μόνο το execution latency.
- **Μεγάλο strict→lenient χάσμα στο Advising** (και στα δύο LLMs) — δείχνει ότι τα LLMs συχνά "καταλαβαίνουν" σωστά την ερώτηση αλλά επιστρέφουν επιπλέον/διαφορετικές στήλες απ' ό,τι το gold SQL.
- **ATIS παραμένει δύσκολο ακόμα και στο lenient** — οφείλεται σε γνωστές ιδιαιτερότητες του πρωτότυπου dataset (hardcoded ημερομηνίες 1991, μη-κυριολεκτικές gold απαντήσεις).
- **Μη-μονότονο easy/medium/hard μοτίβο**: το accuracy στο "medium" είναι χαμηλότερο απ' ό,τι στο "hard" — ένδειξη ότι η αυτόματη (heuristic) κατηγοριοποίηση δυσκολίας δεν αντιστοιχεί τέλεια στην πραγματική δυσκολία μιας ερώτησης για το LLM.

---

## Στατιστική Σημαντικότητα

Σύγκριση GPT vs Qwen με **exact two-sided paired McNemar test** πάνω στις ίδιες 306 ερωτήσεις (κατάλληλο για paired comparison, αφού τα δύο μοντέλα αξιολογούνται στο ίδιο ακριβώς δείγμα):

| Μετρική | GPT-only correct | Qwen-only correct | p-value | Σημαντικό (α=0.05); |
|---|---:|---:|---:|:---:|
| Strict (χωρίς trivial matches) | 18 | 8 | 0.0755 | **Όχι** |
| Lenient | 43 | 11 | 0.000014 | **Ναι** |

**Ερμηνεία:** Με την αυστηρή μετρική, η υπεροχή του GPT δεν μπορεί να θεωρηθεί στατιστικά αποδεδειγμένη σε αυτό το μέγεθος δείγματος. Με την πιο ανεκτική (lenient) μετρική, η υπεροχή του GPT είναι στατιστικά πολύ ισχυρή. Αυτή η αντίθεση δείχνει πόσο ουσιαστικά επηρεάζει η επιλογή μετρικής τα τελικά συμπεράσματα — σημαντικό μεθοδολογικό εύρημα.

---

## Γνωστοί Περιορισμοί

- **Zero-shot / ελάχιστο few-shot**: μόνο 1 παράδειγμα ανά schema· περισσότερα παραδείγματα πιθανότατα θα βελτίωναν σημαντικά το accuracy.
- **Strict metric ευαίσθητο σε επιπλέον στήλες**: ένα σημασιολογικά σωστό SQL μπορεί να "αποτύχει" αν επιστρέφει περισσότερη πληροφορία απ' όσο ζητήθηκε — γι' αυτό αναφέρουμε πάντα strict *και* lenient.
- **`trivial_empty_match`**: δύο ερωτήματα που και τα δύο επιστρέφουν 0 γραμμές (για εντελώς διαφορετικούς/λάθος λόγους) θα μπορούσαν ψευδώς να μετρηθούν ως "ίδια" — το εντοπίζουμε ρητά και το εξαιρούμε από το headline accuracy.
- **Ιδιαιτερότητες πρωτότυπων datasets**: το ATIS/Geography/Advising προέρχονται από ένα ερευνητικό corpus δεκαετιών, με ασυνέπειες case-sensitivity στα table names και ορισμένες gold απαντήσεις που δεν απαντούν κυριολεκτικά στην ερώτηση.
- **Μέγεθος δείγματος στα custom datasets**: 2-15 ερωτήσεις έκαστο — πολύ μικρό για στατιστικά αξιόπιστα ποσοστά μεμονωμένα (τα ευρεία confidence intervals στο σχετικό γράφημα το δείχνουν ξεκάθαρα).
- **Heuristic difficulty labeling**: η αυτόματη κατηγοριοποίηση δυσκολίας δεν είναι τέλεια (βλ. μη-μονότονο easy/medium/hard μοτίβο παραπάνω).

---

## Κατάσταση Παραδοτέων

| Κομμάτι | Κατάσταση |
|---|---|
| Environment setup | ✅ Ολοκληρωμένο |
| Data preparation | ✅ Ολοκληρωμένο |
| Evaluation pipeline + πειράματα (4 combos) | ✅ Ολοκληρωμένο |
| Στατιστική ανάλυση + γραφήματα | ✅ Ολοκληρωμένο |
| Τελική γραπτή αναφορά | ✅ Ολοκληρωμένο — βλ. [`docs/LLMSQL3_Final_Report.docx`](docs/LLMSQL3_Final_Report.docx) |

---

## Άδεια χρήσης

Βλ. [LICENSE](LICENSE).