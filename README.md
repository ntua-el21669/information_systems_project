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

Στο σημείο αυτό τα containers τρέχουν αλλά είναι **κενά** — δεν περιέχουν ακόμα κανένα table ή δεδομένα.

### Φόρτωση δεδομένων στις βάσεις

Κάθε ένα από τα 3 datasets (`geography`, `atis`, `advising`) πρέπει να δημιουργηθεί ως ξεχωριστό database **και στις δύο** βάσεις (MySQL, MariaDB), και να φορτωθεί με το αντίστοιχο `-db.sql` αρχείο του από το `data/raw/<dataset>/`.

**Βήμα 1 — Δημιούργησε τα 3 databases και στις δύο βάσεις:**

```bash
docker exec -i mysql-db mysql -uroot -p1234 -e "CREATE DATABASE geography; CREATE DATABASE atis; CREATE DATABASE advising;"
docker exec -i mariadb-db mariadb -uroot -p1234 -e "CREATE DATABASE geography; CREATE DATABASE atis; CREATE DATABASE advising;"
```

**Βήμα 2 — Φόρτωσε το κάθε `-db.sql` αρχείο μέσα στο αντίστοιχο database, και στις δύο βάσεις.** Σημείωση: το MariaDB image χρησιμοποιεί την εντολή `mariadb` (όχι `mysql`) ως client.

Σε **Windows PowerShell**:
```powershell
Get-Content "data\raw\geography\geography-db.sql" | docker exec -i mysql-db mysql -uroot -p1234 geography
Get-Content "data\raw\geography\geography-db.sql" | docker exec -i mariadb-db mariadb -uroot -p1234 geography

Get-Content "data\raw\atis\atis-db.sql" | docker exec -i mysql-db mysql -uroot -p1234 atis
Get-Content "data\raw\atis\atis-db.sql" | docker exec -i mariadb-db mariadb -uroot -p1234 atis

Get-Content "data\raw\advising\advising-db.sql" | docker exec -i mysql-db mysql -uroot -p1234 advising
Get-Content "data\raw\advising\advising-db.sql" | docker exec -i mariadb-db mariadb -uroot -p1234 advising
```

Σε **macOS/Linux (bash)**, αντικατέστησε το `Get-Content "..." |` με `cat ... |`:
```bash
cat data/raw/geography/geography-db.sql | docker exec -i mysql-db mysql -uroot -p1234 geography
```

**Βήμα 3 — Επιβεβαίωση.** Μετά τη φόρτωση, κάθε database πρέπει να έχει τα εξής tables:

| Database | Πλήθος tables | Ενδεικτικά ονόματα |
|---|---|---|
| `geography` | 7 | `state`, `city`, `river`, `mountain`, `lake`, `border_info`, `highlow` |
| `atis` | ~25 | `flight`, `airport`, `city`, `fare`, κ.ά. |
| `advising` | 15 | `COURSE`, `STUDENT`, `PROGRAM`, κ.ά. (σημ.: εδώ τα table names είναι κεφαλαία) |

Γρήγορος έλεγχος από τη γραμμή εντολών:
```bash
docker exec -it mysql-db mysql -uroot -p1234 geography -e "SHOW TABLES;"
```

Ή μέσω Python:
```bash
python -c "import sys; sys.path.insert(0,'src'); from db_executor import connect, MYSQL_CONFIG, get_schema_description; conn = connect(database='geography', **MYSQL_CONFIG); print(get_schema_description(conn, 'geography')); conn.close()"
```

⚠️ **Σημαντικό:** τα table names είναι case-sensitive στο Linux MySQL/MariaDB. Το `geography`/`atis` έχουν πεζά table names, ενώ το `advising` έχει κεφαλαία — αυτό χειρίζεται ήδη αυτόματα ο κώδικας (`db_executor.normalize_table_case`), δεν χρειάζεται καμία χειροκίνητη ενέργεια.

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

Το Qwen2.5-Coder-7B-Instruct (4-bit quantized) τρέχει ξεχωριστά σε Google Colab με δωρεάν T4 GPU, αφού δεν υπάρχει τοπική πρόσβαση σε GPU. Το notebook (`notebooks/qwen_text2sql_setup.ipynb`) φορτώνει το μοντέλο και εκθέτει μια `generate_sql_qwen()` function με το ίδιο interface όπως το GPT-side.

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
│   ├── run_experiment.py          # Δειγματοληψία (build_full_sample) + πλήρες GPT run
│   ├── export_sample_for_qwen.py  # Εξαγωγή του ΙΔΙΟΥ δείγματος (+ schema) για το Colab
│   ├── score_qwen_results.py      # Βαθμολόγηση αποτελεσμάτων Qwen (μετά το Colab)
│   ├── rescore_against_db.py      # Επανα-εκτέλεση ήδη-παραγόμενου SQL σε άλλη βάση
│   └── analyze_results.py         # Στατιστική ανάλυση (Wilson CI, McNemar test) + γραφήματα
├── notebooks/
│   └── qwen_text2sql_setup.ipynb  # Colab notebook: φόρτωση Qwen (4-bit) + batch inference
├── data/
│   ├── raw/                       # Ωμά datasets (geography, atis, advising: .json + -db.sql)
│   ├── processed/                 # Καθαρά CSV (ανά dataset + ενοποιημένο all_datasets_combined.csv)
│   └── results/
│       ├── results_*.csv          # Τελικά αποτελέσματα αξιολόγησης (4 combos)
│       ├── sample_for_qwen.csv    # Το δείγμα ερωτήσεων (331) με schema, όπως εξήχθη για το Colab
│       ├── qwen_results_raw.csv   # Ωμή έξοδος Qwen από το Colab, πριν το scoring
│       └── analysis/              # Στατιστική ανάλυση + γραφήματα
│           ├── accuracy_overall.svg
│           ├── accuracy_by_dataset.svg
│           ├── accuracy_by_difficulty.svg
│           ├── metric_summary.csv
│           ├── paired_mcnemar_test.csv
│           └── statistical_summary.md
├── docs/
│   └── LLMSQL3_Report.docx        # Τελική γραπτή αναφορά (IEEE format)
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

**Σύνολο: 10,575 (ερώτηση, SQL) pairs**, με αυτόματο labeling δυσκολίας (easy/medium/hard) βάσει heuristic πάνω στην πολυπλοκότητα του SQL.

### Πώς να αναπαράγεις το data preparation

```bash
python src/data_loader.py        # μετατρέπει geography/atis/advising .json -> .csv
python src/custom_queries.py     # δημιουργεί τα 31 custom queries
python src/summary.py            # ενοποιεί τα πάντα -> all_datasets_combined.csv
```

---

## Evaluation Pipeline

### Βασικές τεχνικές αποφάσεις

- **Execution accuracy** αντί για string matching.
- **Δύο μετρικές ανά ερώτηση**: *strict* (ίδιες στήλες/τιμές ακριβώς) και *lenient* (επιτρέπει επιπλέον στήλες, αρκεί να περιέχουν όλες τις σωστές τιμές).
- **`trivial_empty_match` flag**: αν gold και generated SQL επιστρέφουν *και τα δύο* 0 γραμμές, δεν μετράμε ψευδώς ως "σωστό".
- **Unscoreable items**: αποκλείονται **μόνο** ερωτήσεις όπου το ίδιο το gold SQL αποτυγχάνει να εκτελεστεί· ερωτήσεις όπου το gold νόμιμα επιστρέφει 0 γραμμές **παραμένουν** και βαθμολογούνται κανονικά.
- **Αυτόματη παραγωγή schema description** από το `information_schema` της βάσης.
- **Ένα few-shot παράδειγμα ανά schema** μέσα στο prompt.
- **Δειγματοληψία (`build_full_sample`)**: στρωματοποιημένο δείγμα ~300 ερωτήσεων από τα 3 δημόσια datasets (by dataset × difficulty), **συν όλα τα 31 custom queries επιπλέον** — ώστε τα custom queries (ρητή απαίτηση της εκφώνησης) να έχουν πάντα πλήρη αντιπροσώπευση, ανεξάρτητα από το πόσο μικρά είναι σε σχέση με τα δημόσια datasets. Τελικό μέγεθος δείγματος: **331 ερωτήσεις**.

### Πώς να τρέξεις πλήρες evaluation (και τα 4 combos)

```bash
# 1. GPT x MySQL (πραγματική κλήση API)
python src/run_experiment.py

# 2. GPT x MariaDB (reuse του ίδιου SQL, καμία νέα κλήση API)
python src/rescore_against_db.py data/results/results_gpt-4o-mini_mysql.csv mariadb

# 3. Εξαγωγή του ΙΔΙΟΥ δείγματος για Qwen (με schema + few-shot example)
python src/export_sample_for_qwen.py

# 4. Ανέβασε το data/results/sample_for_qwen.csv στο notebooks/qwen_text2sql_setup.ipynb
#    (Google Colab), τρέξε το Qwen (batch mode), κατέβασε ως data/results/qwen_results_raw.csv

# 5. Qwen x MySQL
python src/score_qwen_results.py

# 6. Qwen x MariaDB (reuse, καμία νέα κλήση Colab)
python src/rescore_against_db.py data/results/results_qwen2.5-coder-7b_mysql.csv mariadb

# 7. Στατιστική ανάλυση + γραφήματα (και τα 4 combos)
python src/analyze_results.py
```

**Σημείωση:** μόνο **2 πραγματικές γεννήσεις SQL** χρειάζονται (1 GPT API pass + 1 Qwen Colab pass) για να καλυφθούν και τα 4 combos, αφού το schema είναι δομικά πανομοιότυπο σε MySQL/MariaDB.

---

## Αποτελέσματα

Δείγμα: **331 ερωτήσεις** (~300 στρωματοποιημένες από τα 3 δημόσια datasets + όλα τα 31 custom queries), από τις οποίες **325 (MySQL) / 328 (MariaDB)** είναι scoreable. Πλήρη στατιστικά στο [`data/results/analysis/statistical_summary.md`](data/results/analysis/statistical_summary.md).

### Overall execution accuracy (95% Wilson confidence intervals, N=325)

| Μοντέλο | Μετρική | Αποτέλεσμα | 95% CI |
|---|---|---:|---:|
| GPT-4o-mini | Strict | 50/325 (15.4%) | 11.9%–19.7% |
| GPT-4o-mini | Lenient | 61/325 (18.8%) | 14.9%–23.4% |
| Qwen2.5-Coder-7B-Instruct | Strict | 36/325 (11.1%) | 8.1%–15.0% |
| Qwen2.5-Coder-7B-Instruct | Lenient | 44/325 (13.5%) | 10.2%–17.7% |

### Λοιπές μετρικές

| Metric | GPT×MySQL | GPT×MariaDB | Qwen×MySQL | Qwen×MariaDB |
|---|---|---|---|---|
| Strict accuracy | 15.4% (50/325) | 15.2% (50/328) | 11.1% (36/325) | 11.0% (36/328) |
| Lenient accuracy | 18.8% (61/325) | 18.6% (61/328) | 13.5% (44/325) | 13.4% (44/328) |
| Μέσο generation latency | 1.08s | 1.08s | 4.87s | 4.87s |
| Execution errors (syntax) | 44/325 (13.5%) | 44/328 (13.4%) | 102/325 (31.4%) | 104/328 (31.7%) |

### Accuracy ανά dataset (strict / lenient, MySQL)

| Dataset | n | GPT-4o-mini | Qwen2.5-Coder-7B |
|---|---|---|---|
| Geography | 25 | 64.0% / 68.0% | 64.0% / 64.0% |
| Advising | 122 | 11.5% / 12.3% | 5.7% / 7.4% |
| ATIS | 148 | 8.1% / 12.8% | 2.7% / 5.4% |
| Custom — Geography | 15 | 46.7% / 53.3% | 40.0% / 53.3% |
| Custom — ATIS | 7 | 14.3% / 14.3% | 14.3% / 14.3% |
| Custom — Advising | 8 | 0.0% / 12.5% | 25.0% / 25.0% |

Βλ. γραφήματα: [`accuracy_by_dataset.svg`](data/results/analysis/accuracy_by_dataset.svg), [`accuracy_by_difficulty.svg`](data/results/analysis/accuracy_by_difficulty.svg).

### Βασικά ευρήματα

- **GPT νικά το Qwen** σε accuracy (strict και lenient), με στατιστικά σημαντική διαφορά και στις δύο μετρικές, και είναι ~4.5x πιο γρήγορο.
- **Qwen κάνει σχεδόν διπλάσια syntax errors** — αναμενόμενο για ένα μικρότερο, τοπικά τρέχον, quantized μοντέλο.
- **Το RDBMS (MySQL vs MariaDB) δεν επηρεάζει ουσιαστικά το accuracy** — επηρεάζει ελαφρώς μόνο ποια συγκεκριμένα gold queries εκτελούνται (dialect-level διαφορές) και το execution latency.
- **Geography είναι το πιο "εύκολο" dataset**, ενώ το ATIS το πιο δύσκολο — λογικό μοτίβο.
- **Μη-μονότονο easy/medium/hard μοτίβο**: ένδειξη ότι η αυτόματη κατηγοριοποίηση δυσκολίας δεν αντιστοιχεί τέλεια στην πραγματική δυσκολία μιας ερώτησης.

---

## Στατιστική Σημαντικότητα

Σύγκριση GPT vs Qwen με **exact two-sided paired McNemar test** πάνω στις ίδιες 325 scoreable ερωτήσεις:

| Μετρική | GPT-only correct | Qwen-only correct | p-value | Σημαντικό (α=0.05); |
|---|---:|---:|---:|:---:|
| Strict | 22 | 8 | 0.0161 | **Ναι** |
| Lenient | 26 | 9 | 0.0060 | **Ναι** |

**Ερμηνεία:** Και με τις δύο μετρικές, η υπεροχή του GPT-4o-mini έναντι του Qwen2.5-Coder-7B είναι στατιστικά σημαντική στο επίπεδο α=0.05.

---

## Γνωστοί Περιορισμοί

- **Zero-shot / ελάχιστο few-shot**: μόνο 1 παράδειγμα ανά schema.
- **Lenient metric αγνοεί ταυτότητα στηλών**: ελέγχει αν οι τιμές του gold εμφανίζονται κάπου στο generated αποτέλεσμα, χωρίς να λαμβάνει υπόψη από ποια στήλη προέρχεται η κάθε τιμή — σε σπάνιες περιπτώσεις θα μπορούσε να δώσει ψευδώς θετικό αποτέλεσμα.
- **Ιδιαιτερότητες πρωτότυπων datasets**: το ATIS/Geography/Advising έχουν ορισμένες gold απαντήσεις που δεν απαντούν κυριολεκτικά στην ερώτηση (π.χ. hardcoded ιστορικές ημερομηνίες στο ATIS).
- **Heuristic difficulty labeling**: δεν αντιστοιχεί πάντα τέλεια στην πραγματική δυσκολία μιας ερώτησης για ένα LLM.
- **Μέγεθος δείγματος στα custom datasets**: 7-15 ερωτήσεις ανά κατηγορία — αρκετό για ενδεικτική εικόνα, με αντίστοιχα ευρύτερα confidence intervals.
- **Single-question duplication**: 1 ζευγάρι πανομοιότυπων ερωτήσεων (ATIS, διαφορετικά splits του πρωτότυπου corpus) εντοπίστηκε στο δείγμα — αμελητέα επίπτωση (330 αντί 331 μοναδικές ερωτήσεις).


## Άδεια χρήσης

Βλ. [LICENSE](LICENSE).
