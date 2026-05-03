# DataProc Ingestion Performance Improvements

## File: `scripts/dataproc_ingest_and_sample_combined.py`

---

## Changes

### 1. Stream CSV instead of downloading the full file

**What we changed:**
Used `stream=True` on the CSV request and stopped reading after collecting enough lines for 5 rows.

**Why:**
The original code downloaded the entire CSV file into memory before taking just 5 rows. Some public datasets have CSVs that are gigabytes in size. This was the biggest bottleneck — downloading hundreds of MB just to keep a few KB of data.

**Impact:**
- No change in data (same 5 rows)
- Dramatically reduces network transfer per dataset
- Reduces memory usage per worker

---

### 2. Use Socrata JSON API for NYC sample rows instead of CSV

**What we changed:**
For NYC Open Data datasets, replaced CSV download with a direct JSON API call:
`https://data.cityofnewyork.us/resource/{dataset_id}.json?$limit=5`

**Why:**
NYC Open Data (Socrata) has a JSON API that returns exactly N rows on request. The original code was downloading the full CSV (potentially millions of rows) just to get 5. The JSON API is purpose-built for this and returns only what we ask for.

**Impact:**
- No change in data (same 5 rows, same columns)
- Much faster for NYC datasets specifically
- Less network load on the cluster

---

### 3. Increase number of Spark partitions

**What we changed:**
Changed partition count from `max(8, ceil(200/10))` = 20 to `max(40, ceil(200/5))` = 40.

**Why:**
With 3 worker nodes on the NYU DataProc cluster, 20 partitions means each worker handles ~7 partitions sequentially. Increasing to 40 partitions gives finer-grained parallelism and keeps all workers busier throughout the job.

**Impact:**
- No change in data
- Better utilization of the cluster
- Reduces overall job time

---

### 4. Reduce request timeout from 90s to 30s

**What we changed:**
Lowered the `timeout` parameter on all HTTP requests from 90 seconds to 30 seconds.

**Why:**
The original 90-second timeout meant a single unresponsive dataset URL could stall a worker for 90 seconds. Datasets that don't respond within 30 seconds are almost always broken URLs — they fail either way and get recorded as errors with empty sample rows.

**Impact:**
- No loss of real data (broken URLs were already failing)
- Workers fail faster on bad URLs and move on
- Reduces worst-case time significantly

---

---

### 5. Explicit schema for `createDataFrame` (Bug Fix in v2)

**What we changed:**
Added an explicit `StructType` schema when calling `spark.createDataFrame(enriched_rdd, schema=schema)` instead of relying on Spark's schema inference.

**Why:**
When `process_partition` catches an exception mid-processing, it returns an error record where most fields are `None`. If Spark samples these rows during schema inference, it sees all `None`s for a field and cannot determine whether it's a string, int, or any other type — throwing:
```
PySparkValueError: [CANNOT_DETERMINE_TYPE] Some of types cannot be determined after inferring.
```
This caused the job to fail after all the data had already been fetched.

**Impact:**
- Fixes a crash that would occur at the `createDataFrame` step
- No change in data — all fields are already strings or None
- Both the original script and v2 had this bug; v2 is the fixed version

---

## Summary

| Change | Data Impact | Speed Impact |
|---|---|---|
| Stream CSV | None | High |
| NYC JSON API | None | High |
| More partitions | None | Medium |
| Lower timeout | None | Medium |
| Explicit schema | None (bug fix) | None |

The v2 script includes all performance optimizations and the schema inference bug fix.
