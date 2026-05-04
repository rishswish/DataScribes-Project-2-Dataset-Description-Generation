# DataScribes — Scalable Dataset Description Generation

Automatically generate high-quality, user-focused descriptions for public datasets using the Claude Sonnet LLM. The pipeline ingests metadata from NYC Open Data and Data.gov, profiles the datasets, builds structured prompts, and calls the Anthropic API at scale using PySpark on NYU's DataProc cluster.

---

## Results

| Metric | Value |
|---|---|
| Datasets processed | 200 (100 NYC Open Data + 100 Data.gov) |
| Successful descriptions | 196 / 200 (98%) |
| LLM model | claude-sonnet-4-5 |
| Avg generated length | ~102 words, ~4.7 sentences |
| Novelty score | 0.67 (67% new vocabulary vs. original) |
| Title coverage | 99% |
| ROUGE-1 | 0.365 |
| METEOR | 0.292 |
| BERTScore F1 | 0.860 |
| NDCG@10 improvement | +0.012 over original descriptions |

---

## Pipeline Overview

```
[NYC Open Data API]     [Data.gov DCAT API]
        │                       │
        └──────────┬────────────┘
                   ▼
     Stage 1: Ingestion & Sampling
     dataproc_ingest_and_sample_v2.py
     → combined_metadata_with_samples_v2.parquet (HDFS)
                   │
                   ▼
     Stage 2: Profiling
     dataproc_profile_combined.py
     → combined_profiles_spark_v2.parquet (HDFS)
                   │
                   ▼
     Stage 3: LLM Description Generation
     dataproc_generate_descriptions.py
     → generated_descriptions_sonnet.parquet (HDFS)
                   │
                   ▼
     Stage 4: Evaluation
     evaluate_llm_descriptions.py  (automatic metrics)
     evaluate_full.py              (ROUGE, METEOR, BERTScore, NDCG)
```

---

## Repository Structure

```
.
├── configs/
│   └── project_config.yaml          # Storage paths, execution engine config
│
├── ingestion/
│   ├── schema.py                    # Canonical PySpark StructType schema
│   ├── nyc_open_data.py             # Local NYC Open Data ingestion (pilot)
│   └── data_gov.py                  # Local Data.gov ingestion (pilot)
│
├── profiling/
│   ├── profile_nyc_metadata.py      # Local pandas profiler for NYC
│   └── profile_data_gov_metadata.py # Local pandas profiler for Data.gov
│
├── llm/
│   ├── build_llm_input.py           # Builds structured LLM input JSON
│   ├── generate_descriptions.py     # Template baseline v1 (no LLM)
│   └── generate_descriptions_v2.py  # Template baseline v2 (no LLM)
│
├── scripts/
│   ├── dataproc_ingest_and_sample_v2.py      # Stage 1 — DataProc ingestion
│   ├── dataproc_profile_combined.py          # Stage 2 — DataProc profiling
│   ├── dataproc_generate_descriptions.py     # Stage 3 — LLM generation
│   ├── build_final_project_table.py          # Joins profiles + descriptions
│   ├── build_report_summary.py               # Summary stats CSV/XLSX
│   ├── build_manual_eval_sample.py           # Manual scoring spreadsheet
│   └── generate_project_doc.py               # Generates project Word doc
│
├── evaluation/
│   ├── evaluate_llm_descriptions.py  # Automatic metrics (novelty, coverage)
│   ├── evaluate_full.py              # ROUGE, METEOR, BERTScore, NDCG
│   ├── compare_descriptions.py       # Template v1 vs v2 for NYC
│   └── compare_data_gov_descriptions.py  # Template v2 for Data.gov
│
├── data/
│   ├── metadata/
│   │   └── combined_metadata_with_samples.parquet  # 20-row pilot (local)
│   ├── descriptions/
│   │   └── generated_descriptions_sonnet.parquet/  # 200-row output (from HDFS)
│   ├── evaluation_results.csv          # Per-dataset automatic metrics
│   ├── evaluation_summary.csv          # Aggregated automatic metrics
│   ├── text_similarity_results.csv     # ROUGE / METEOR / BERTScore per dataset
│   ├── retrieval_results.csv           # NDCG@5 and NDCG@10 per query
│   ├── full_evaluation_summary.csv     # NLP metrics summary table
│   ├── manual_evaluation_sample.xlsx   # 10-dataset hand-scoring spreadsheet
│   └── DataScribes_Project2_File_Reference.docx  # Full project reference doc
│
├── docs/
│   └── dataproc_performance_improvements.md  # Optimization decisions
│
└── requirements.txt
```

---

## HDFS Files (NYU DataProc Cluster)

These files live on the cluster under `hdfs:///user/km6579_nyu_edu/data/` and are not in this repository.

| Path | Description |
|---|---|
| `metadata/combined_metadata_with_samples_v2.parquet` | 200 datasets with metadata + 5 sample rows each |
| `profiles/combined_profiles_spark_v2.parquet` | Same 200 rows + column-level profiling stats |
| `descriptions/generated_descriptions_sonnet.parquet` | LLM-generated descriptions (10 Spark part files) |

---

## How to Run

### Prerequisites

- Python 3.9+
- Anaconda (for local evaluation scripts)
- Access to NYU DataProc cluster
- Anthropic API key

### Install dependencies (local)

```bash
pip install -r requirements.txt
pip install rouge-score bert-score nltk
```

### Upload scripts to the cluster

Open the DataProc master node in SSH-in-browser (via Google Cloud Console). Use the **Upload File** button in the top-right corner to upload each script from your local machine. Once uploaded, move them into the scripts folder:

```bash
mv dataproc_ingest_and_sample_v2.py ~/scripts/
mv dataproc_profile_combined.py ~/scripts/
mv dataproc_generate_descriptions.py ~/scripts/
```

### Stage 1 — Ingestion & Sampling (DataProc)

```bash
# Run from inside the master node
spark-submit ~/scripts/dataproc_ingest_and_sample_v2.py
```

Fetches 100 NYC + 100 Data.gov datasets in parallel across 40 Spark partitions. Uses the Socrata JSON API for NYC sample rows and streams CSVs for Data.gov to avoid downloading full files.

Output is written to HDFS automatically — no manual HDFS copy needed.

### Stage 2 — Profiling (DataProc)

```bash
spark-submit ~/scripts/dataproc_profile_combined.py
```

Reads the metadata parquet from HDFS, applies Spark UDFs to compute per-column statistics (type inference, missing counts, example values).

### Stage 3 — LLM Generation (DataProc)

The `anthropic` package must be bundled and shipped to worker nodes before running this stage. `pip install anthropic` on the master node only installs it there — the worker nodes (`nyu-dataproc-w-0`, `nyu-dataproc-w-1`) are separate machines and would throw `ModuleNotFoundError`. NYU's cluster does not allow direct SSH into workers, so the fix is to zip the package and pass it via `--archives`, which tells Spark to ship it to every worker automatically.

```bash
# 1. Install the package into a local folder
pip install anthropic -t ~/anthropic_pkg

# 2. Zip it so Spark can distribute it
cd ~
zip -r anthropic_pkg.zip anthropic_pkg/

# 3. Set your Anthropic API key
export ANTHROPIC_API_KEY=<your-key>

# 4. Submit — --archives ships the zip to every worker and extracts it as "anthropic_pkg/"
spark-submit \
  --archives ~/anthropic_pkg.zip#anthropic_pkg \
  ~/scripts/dataproc_generate_descriptions.py
```

Inside `generate_description_for_partition()`, `sys.path.insert(0, "anthropic_pkg")` tells each worker where to find the extracted package (see [scripts/dataproc_generate_descriptions.py](scripts/dataproc_generate_descriptions.py)).

Calls `claude-sonnet-4-5` via the Anthropic API using `mapPartitions` across 10 partitions. Builds one structured prompt per dataset from title, description, keywords, column profiles, and a sample row.

### Download output from HDFS (after Stage 3)

```bash
# On the master node — copy parquet output from HDFS to local filesystem
hdfs dfs -get /user/km6579_nyu_edu/data/descriptions/generated_descriptions_sonnet.parquet ~/

# Then from your local machine — copy from master node to local
gcloud compute scp --recurse <master-node>:~/generated_descriptions_sonnet.parquet ./data/descriptions/ --zone=<zone>
```

### Stage 4 — Evaluation (local)

```bash
# Automatic metrics
python evaluation/evaluate_llm_descriptions.py

# ROUGE, METEOR, BERTScore, NDCG
python evaluation/evaluate_full.py
```

Reads `generated_descriptions_all_200.csv` from `~/Downloads/` or `data/`.

---

## The LLM Prompt

Each dataset prompt is assembled by `build_prompt()` in `scripts/dataproc_generate_descriptions.py`:

```
You are a data catalog assistant. Write a clear, concise description
(3-5 sentences) for the following dataset.
The description should help a data analyst quickly understand what
the dataset contains, what it can be used for, and any notable
characteristics. Do not copy the original description verbatim.
Be specific and informative.

Dataset Title: {title}
Source: {source}
Original Description: {original_description}    ← capped at 500 chars
Keywords: {kw1}, {kw2}, ...                     ← up to 8 keywords
Columns (N total): {col1}, {col2}, ...          ← up to 12 column names
Column Profiles:
  - {col_name} (numeric_like/text_like): e.g. {val1}, {val2}
                                                ← up to 6 columns
Sample row: {first_row_json}                    ← capped at 300 chars

Description:
```

---

## Evaluation Methods

### Automatic Metrics (`evaluate_llm_descriptions.py`)

| Metric | What it measures |
|---|---|
| Success rate | % of datasets with a non-null generated description |
| Word count | Original vs. generated description length |
| Sentence count | Whether the LLM followed the 3-5 sentence instruction |
| Novelty score | Fraction of vocabulary in generated text not present in original |
| Title coverage | Whether the generated description references the dataset subject |

### Text Similarity (`evaluate_full.py`)

Treats the original human-written description as the reference and scores the generated description against it.

| Metric | What it measures |
|---|---|
| ROUGE-1/2/L | Word and phrase overlap between generated and original |
| METEOR | Like ROUGE but accounts for synonyms and stemming |
| BERTScore F1 | Semantic similarity using RoBERTa-large embeddings |

### Retrieval Evaluation (`evaluate_full.py`)

Simulates dataset search. Builds a TF-IDF index over all 196 datasets using either the original or the generated descriptions, then runs 8 realistic search queries and measures ranking quality with NDCG@5 and NDCG@10.

| Query | NDCG@10 (original) | NDCG@10 (generated) |
|---|---|---|
| NYC taxi trip data | 1.000 | 1.000 |
| Crime incident reports | 0.910 | 0.990 |
| Weather wind speed dataset | 0.982 | 1.000 |
| Restaurant food inspection | 1.000 | 1.000 |
| Overall average | 0.987 | **0.999** |

### Qualitative Evaluation

`data/manual_evaluation_sample.xlsx` contains 10 datasets (5 NYC + 5 Data.gov) with blank columns for human scoring: `manual_score_readability`, `manual_score_accuracy`, `manual_score_usefulness`, `manual_notes`.

---

## Known Failures

4 datasets failed during LLM generation, all from NYC Open Data:

| Dataset ID | Title | Error |
|---|---|---|
| `6ztr-wgff` | Transit Zones | Rate limit (429) |
| `ek8y-fsqz` | Sea Level Rise Maps (2080s) | Rate limit (429) |
| `rf9r-c4pz` | Sea Level Rise Maps (2100) | Rate limit (429) |
| `vhqf-adkz` | Greater Transit Zone | Prompt too long (218,702 tokens) |

The rate-limit failures can be retried with exponential backoff. The prompt-too-long failure requires truncating the original description before building the prompt.

---

## Data Sources

- **NYC Open Data** — [data.cityofnewyork.us](https://data.cityofnewyork.us) via the Socrata API
- **Data.gov** — [catalog.data.gov](https://catalog.data.gov) via the DCAT catalog API

---

## Performance Optimizations (branch: `k-dataproc-scalling`)

Documented in `docs/dataproc_performance_improvements.md`:

1. **Stream CSV** — stops downloading after 5 rows instead of fetching the full file
2. **Socrata JSON API for NYC** — requests exactly 5 rows via `?$limit=5` instead of CSV download
3. **40 Spark partitions** — up from 20, better utilisation of the 3-node cluster
4. **30s request timeout** — down from 90s, fails fast on broken URLs
5. **Explicit schema** — passes `StructType` to `createDataFrame` to prevent schema inference crash when error rows contain all-None fields

---

## Requirements

```
pyspark==3.5.1
pandas==2.2.2
requests==2.32.3
pyyaml==6.0.2
pyarrow==16.1.0
rouge-score
bert-score
nltk
scikit-learn
```
