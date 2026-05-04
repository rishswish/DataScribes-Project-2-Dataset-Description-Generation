# Full Evaluation Report

## Overview
- Datasets processed: `200`
- Successful generations: `196` (98.0%)
- Failed generations: `4`

## Quality Metrics
| Source | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | BERTScore F1 |
| --- | --- | --- | --- | --- | --- |
| nyc_open_data | 0.3510 | 0.1038 | 0.1980 | 0.2873 | 0.8500 |
| data_gov | 0.3778 | 0.1335 | 0.2302 | 0.2955 | 0.8675 |
| OVERALL | 0.3647 | 0.1190 | 0.2145 | 0.2915 | 0.8590 |

## Retrieval Metrics
| Metric | Original | Generated | Delta |
| --- | --- | --- | --- |
| NDCG@5 | 0.9934 | 1.0000 | +0.0066 |
| NDCG@10 | 0.9865 | 0.9987 | +0.0122 |

## Failure Analysis
- `rate_limit`: 3 dataset(s)
- `prompt_too_long`: 1 dataset(s)

The current failures fall into two operational buckets:
- `rate_limit`: the generation request exceeded Anthropic throughput limits at runtime.
- `prompt_too_long`: the prompt payload exceeded the model token limit for a specific dataset.

These failures do not indicate a parsing bug in the evaluation pipeline; they indicate generation-time robustness issues that should be addressed by retry logic, batching controls, or prompt truncation.

## Notable Retrieval Gains
| Query | Original NDCG@10 | Generated NDCG@10 | Delta |
| --- | --- | --- | --- |
| crime incident reports | 0.9103 | 0.9896 | +0.0793 |
| weather wind speed dataset | 0.9818 | 1.0000 | +0.0182 |
| NYC taxi trip data | 1.0000 | 1.0000 | +0.0000 |

## Qualitative Review Sheet
A manual review spreadsheet was generated at `data/manual_evaluation_sample.xlsx`.
The sample now includes representative high-quality rows, low-quality rows, mid-quality rows, and failed generations so that qualitative scoring covers both strengths and edge cases.

## Output Files
- `data/text_similarity_results.csv`
- `data/retrieval_results.csv`
- `data/full_evaluation_summary.csv`
- `data/failure_analysis.csv`
- `data/manual_evaluation_sample.csv`
- `data/manual_evaluation_sample.xlsx`
