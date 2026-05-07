# 📊 Scalable Dataset Description Generation for Open Data Repositories

**Team DataScribes** | Data Engineering Course Project | NYU Center for Data Science 

<div align="center">
  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python >= 3.10">
    <img src="https://img.shields.io/badge/Apache%20Spark-3.x-E25A1C?style=flat-square&logo=apachespark&logoColor=white" alt="Apache Spark">
    <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-blue?style=flat-square&logo=openai" alt="OpenAI">
    <img src="https://img.shields.io/badge/HDFS-Distributed%20Storage-66CCFF?style=flat-square&logo=apache" alt="HDFS">
    <img src="https://img.shields.io/badge/NYU-HPC%20DataProc-57068C?style=flat-square" alt="NYU HPC">
  </p>
</div>

---

## Overview

This project implements a scalable, end-to-end pipeline for **automatically generating high-quality textual descriptions** for datasets from open data repositories. It combines distributed data processing using **Apache Spark** on the NYU HPC DataProc cluster with **LLM-based semantic analysis** via GPT-4o-mini to produce readable, informative dataset summaries.

The pipeline addresses a key challenge in open data discoverability: many datasets lack clear or complete descriptions, making them difficult to find and understand. By automating description generation at scale, this system improves dataset metadata quality and discoverability.

---

## Data Sources

| Source | API | Datasets Fetched |
|--------|-----|-----------------|
| [NYC Open Data](https://opendata.cityofnewyork.us/) | Socrata Open Data API (SODA) | 200 |
| [Data.gov](https://data.gov/) | CKAN Catalog API | 200 |
| **Total** | | **400** |

---

## System Architecture

The pipeline consists of 5 main stages:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Step 1         │    │  Step 2         │    │  Step 3         │
│  Metadata       │───▶│  CSV Sample     │───▶│  Data           │
│  Ingestion      │    │  Fetching       │    │  Preparation    │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Step 5         │    │  Step 4         │    │                 │
│  Evaluation     │◀───│  LLM Description│◀───│  Prepared Data  │
│  & Analysis     │    │  Generation     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Infrastructure
- **Cluster**: NYU HPC DataProc (YARN)
  - Master node: `nyu-dataproc-m` (n1-standard-32)
  - Worker nodes: 2× `nyu-dataproc-w` (n1-standard-16)
  - Secondary worker: 1× preemptible (n1-standard-16)
- **Storage**: Hadoop Distributed File System (HDFS)
- **Processing**: Apache Spark (PySpark)
- **LLM**: GPT-4o-mini via OpenAI API

---

## Pipeline Configuration

```python
NYC_LIMIT           = 200       # Datasets from NYC Open Data
DATA_GOV_LIMIT      = 200       # Datasets from Data.gov
MODEL_NAME          = "gpt-4o-mini"
MAX_SAMPLE_ROWS     = 20        # CSV rows sampled per dataset
REQUEST_TIMEOUT     = 40        # HTTP timeout (seconds)
MAX_ELAPSED_SECONDS = 60        # Max streaming time per dataset
MAX_CHARS           = 12000     # Max characters sent to LLM
MAX_LINES           = 20        # Max lines sent to LLM
```

---

## Results

### Pipeline Output
| Metric | Value |
|--------|-------|
| Datasets requested | 400 |
| Datasets successfully described | 232 |
| NYC Open Data success rate | 95% |
| Data.gov success rate | 45% (many lack CSV links) |

### Evaluation Results

#### Text Similarity Metrics
Evaluated on 203 datasets with both original and generated descriptions (n = 203).

| Metric | Generated vs Original | Description |
|--------|----------------------|-------------|
| ROUGE-1 | 0.2438 | Unigram overlap with original descriptions |
| ROUGE-2 | 0.0436 | Bigram overlap |
| ROUGE-L | 0.1435 | Longest common subsequence |
| METEOR | 0.1636 | Overlap accounting for synonyms |

> Low ROUGE/METEOR scores are expected: LLM-generated text naturally uses different phrasing than human-written originals. These metrics penalize paraphrasing even when the meaning is equivalent.

#### Retrieval Evaluation (NDCG@10)
Simulates a dataset search engine scenario across 8 test queries.

| | Generated | Search Desc | Original |
|---|---|---|---|
| Average NDCG@10 | 0.1502 | 0.1938 | 0.1691 |

Search descriptions score highest as expected — they are keyword-rich by design. Generated (user-facing) descriptions are competitive with originals overall and outperform them on some queries (e.g., bicycle/pedestrian counts: 0.1324 vs 0.0454).

#### AutoDDG Intrinsic Quality Evaluation
GPT-4o-mini scores each generated description independently on three dimensions (no reference needed). Evaluated across all 207 successfully generated descriptions.

| Dimension | Avg Score (out of 10) |
|-----------|----------------------|
| Completeness | 7.26 |
| Conciseness | 8.13 |
| Readability | 7.99 |
| **Overall** | **7.79** |

#### AutoDDG Pairwise Evaluation
GPT-4o-mini acts as a judge, directly comparing generated descriptions head-to-head against human-written originals (201 pairs with substantive original descriptions > 50 characters).

| | Count | Share |
|---|---|---|
| Generated wins | 192 | 95.5% |
| Original wins | 9 | 4.5% |

**Breakdown by source:**

| Source | Generated Wins | Original Wins |
|--------|---------------|---------------|
| NYC Open Data | 131 | 6 |
| Data.gov | 61 | 3 |

#### ELO Ratings
Tournament-style aggregate performance across all 201 pairwise comparisons (baseline: 1500).

| | ELO Rating |
|---|---|
| Generated descriptions | **1514.6** |
| Original descriptions | 1485.4 |

> Generated descriptions outperform human-written originals in 95.5% of head-to-head comparisons, achieving a higher ELO rating across both sources. The intrinsic scores confirm the descriptions are consistently complete, concise, and readable — including for the many datasets that had no original description at all.

---

## Repository Structure

```
├── notebooks/
│   └── autoddg_pipeline_hpc.ipynb   # Main pipeline notebook
├── scripts/
│   ├── run_fetch_nyc_metadata.py
│   ├── run_build_evaluation_table.py
│   └── run_text_similarity_metrics.py
├── outputs/
│   └── descriptions.parquet          # Generated descriptions
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## How to Run

### Prerequisites
- Access to NYU HPC DataProc cluster
- OpenAI API key
- Python 3.10+

### Setup
```bash
# Clone the repo
git clone https://github.com/rishswish/DataScribes-Project-2-Dataset-Description-Generation.git
cd DataScribes-Project-2-Dataset-Description-Generation

# Install dependencies
pip install -e .
pip install openai pyspark rouge-score bert-score nltk
```

### Running the Pipeline
```bash
# SSH into HPC master node
gcloud compute ssh nyu-dataproc-m --zone=us-central1-f --project=hpc-dataproc-19b8

# Start Jupyter with port forwarding
jupyter notebook --no-browser --port=8888

# In a separate terminal, open tunnel
gcloud compute ssh nyu-dataproc-m --zone=us-central1-f \
  --ssh-flag="-N" --ssh-flag="-L 8888:localhost:8888"
```

Then open `http://localhost:8888` and run `notebooks/autoddg_pipeline_hpc.ipynb`.

---

## Course Information

- **Course**: Big Data 
- **Instructor**: Professor Juliana Freire
- **Section Leader**: Dr. Christos Koutras
- **Team**: DataScribes
- **Members**: Rishabh Patil (rbp5812), Kund Meghani (km6579), Aditya Taware (at6370)

---

## Attribution & Acknowledgements

This project builds upon and was inspired by the **AutoDDG** framework developed by researchers at NYU VIDA Lab:

> **AutoDDG: Automated Dataset Description Generation using Large Language Models**
> Haoxiang Zhang, Yurong Liu, Wei-Lun Hung, Aécio Santos, Juliana Freire
> *Proceedings of the ACM on Management of Data (SIGMOD 2026)*
> arXiv: https://arxiv.org/abs/2502.01050

The AutoDDG library was cloned from [https://github.com/VIDA-NYU/AutoDDG](https://github.com/VIDA-NYU/AutoDDG) and modified to support:
- Large-scale distributed ingestion from multiple open data APIs (NYC Open Data, Data.gov)
- Integration with Apache Spark on NYU HPC DataProc for distributed processing
- HDFS-based intermediate storage across pipeline stages
- Batch evaluation using ROUGE, METEOR, BERTScore, and NDCG metrics

We gratefully acknowledge the AutoDDG authors for their open-source framework which served as the foundation for the description generation component of this pipeline.

```bibtex
@article{autoddg-sigmod2026,
  Author  = {Haoxiang Zhang and Yurong Liu and Wei-Lun Hung and Aécio Santos and Juliana Freire},
  Title   = {AutoDDG: Automated Dataset Description Generation using Large Language Models},
  Journal = {Proceedings of the ACM on Management of Data},
  Year    = {2026},
  Note    = {To appear},
}
```

---

## License

This project is released under the [Apache License 2.0](./LICENSE), consistent with the original AutoDDG repository.
