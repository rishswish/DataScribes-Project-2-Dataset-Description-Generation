# Scalable Dataset Description Generation

## V1 Scope
- Sources:
  - NYC Open Data
  - Data.gov
- Dataset type:
  - Tabular datasets only
- Initial scale:
  - 20 to 50 datasets total
- Compute:
  - PySpark on NYU DataProc
- Storage:
  - raw API responses as JSON
  - normalized metadata as Parquet
  - profiling outputs as Parquet
  - generated descriptions as JSON
- Description type:
  - one user-focused description per dataset
- LLM input:
  - title
  - original description
  - keywords
  - column names
  - profiling summary
  - sampled rows