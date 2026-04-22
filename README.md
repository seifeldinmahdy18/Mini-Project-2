# Mini-Project-2 — Steam Reviews Big Data Analysis

A PySpark-based analysis of **15.4 million Steam reviews (~5 GB)** using the DataFrame API with the Catalyst Optimizer.

**Dataset:** [Steam Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/forgemaster/steam-reviews-dataset)

---

## Dataset

### Overview

| Property | Value |
|---|---|
| Records | 15,400,000+ |
| Raw Size | ~5 GB |
| Format | CSV / Parquet |
| Source | Kaggle — `forgemaster/steam-reviews-dataset` |

### Columns

| Column | Type | Description |
|---|---|---|
| `review_id` | Long | Unique identifier for each review |
| `app_id` | Long | Steam application ID of the reviewed game |
| `author_steamid` | Long | Steam ID of the reviewer |
| `language` | String | Language the review was written in (e.g. `english`, `russian`) |
| `review` | String | Full text of the review |
| `timestamp_created` | Long | Unix epoch timestamp of when the review was submitted |
| `recommended` | Boolean | Whether the reviewer recommended the game (`true` / `false`) |
| `votes_helpful` | Integer | Number of users who marked the review as helpful |
| `votes_funny` | Integer | Number of users who marked the review as funny |
| `weighted_vote_score` | Float | Steam's internal helpfulness score (0.0 – 1.0), combining votes and recency |
| `author_playtime_forever` | Integer | Total minutes the reviewer has played the game at the time of writing |

---

## Project Structure

```
MiniProject2/
├── data/                        # Raw and processed dataset files
├── scripts/
│   └── fetch_data.py            # Downloads and extracts the dataset from Kaggle
├── src/
│   ├── dataframe/
│   │   └── main_queries.py      # All 10 DataFrame API queries (this module)
│   ├── rdd/                     # RDD-based processing (separate module)
│   └── sql/                     # SparkSQL queries (separate module)
├── report/                      # Analysis reports and outputs
├── .env                         # KAGGLE_USERNAME and KAGGLE_KEY
└── README.md
```

---

## Setup

### 1. Configure Kaggle credentials

Add your credentials to `.env`:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

### 2. Fetch the dataset

```bash
python scripts/fetch_data.py
```

### 3. Run the DataFrame queries

```bash
spark-submit src/dataframe/main_queries.py
```

---

## The 10 Queries

All queries are implemented in `src/dataframe/main_queries.py` using the PySpark DataFrame API.

| # | Query Type | Description | Key Optimization |
|---|---|---|---|
| 1 | **Complex Filter** | Filter reviews by playtime (≥ 50 hrs), helpfulness score (≥ 0.7), positive recommendation, and review length (≥ 100 chars) to isolate high-quality "Elite" reviewers | Catalyst pushes all predicates to the Parquet row-group reader — unneeded file chunks are never loaded |
| 2 | **Aggregation** | Compute average helpfulness, total review count, average playtime, and total helpful votes per game (minimum 50 reviews) | Partial Hash Aggregate runs on each executor before the shuffle, reducing network bytes by 80–90% |
| 3 | **Multi-Attribute Grouping** | Group by `language` × `recommended` to compare review sentiment and helpfulness across language communities | AQE coalesces small post-shuffle partitions automatically, avoiding empty-task overhead |
| 4 | **Sorting & Ranking** | Rank games by a composite engagement score (`total_reviews × avg_helpfulness × log(avg_playtime + 1)`) and return the top 10 | Catalyst rewrites ORDER BY + LIMIT as `TakeOrderedAndProject`, using a per-executor top-K heap instead of a global sort |
| 5 | **Window Function — Rank** | Assign a rank to every review within its game partition (ordered by helpfulness) and keep only the single most influential review per game | Partitioned window triggers one `Exchange hashpartitioning(app_id)` shuffle; AQE handles skewed game partitions |
| 6 | **Window Function — Moving Average** | Compute a 7-day rolling average of daily review volume per game to surface review trend momentum | Pre-aggregate to daily counts before applying the range window; avoids memory spill from windowing 15.4M raw rows |
| 7 | **Nested / Subquery Logic** | Return elite reviews that belong to games whose average helpfulness exceeds the global average (two-phase filter) | Global average is computed as a scalar subquery in one scan pass and broadcast to the outer filter; no double scan |
| 8 | **Broadcast Join** | Join the 15.4M reviews with a small `app_metadata` lookup table (game name, genre) using an explicit broadcast hint | `F.broadcast()` sends the small table to every executor as a hash table — the metadata side produces zero shuffle bytes |
| 9 | **Partition Pruning** | Filter reviews to 2022–2023 only and aggregate monthly statistics, demonstrating directory-level I/O elimination | When data is written with `.partitionBy("review_year", "review_month")`, Spark opens only the matching subdirectories |
| 10 | **Sort-Merge Join** | Join the 15.4M reviews with a large `user_profiles` table on `author_steamid` to add account-level context | Pre-repartitioning on the join key before the join aligns partitioning so Spark can skip one Exchange phase |

Each query includes:
- A `.explain(True)` call (for optimization-heavy queries) showing the full Catalyst plan
- A `timer()` wrapper that prints wall-clock execution time for every action

---

## Configuration Notes

The `SparkSession` in `main_queries.py` is pre-configured for a 5 GB workload on a cloud cluster:

| Setting | Value | Reason |
|---|---|---|
| `spark.sql.adaptive.enabled` | `true` | Runtime re-optimization (AQE) |
| `spark.sql.shuffle.partitions` | `200` | ~128 MB per partition for 5 GB; scale to 400–800 for larger data |
| `spark.serializer` | Kryo | ~5× faster than Java default for inter-executor transfer |
| `spark.sql.parquet.filterPushdown` | `true` | Pushes WHERE clauses into the Parquet reader |
| `spark.sql.autoBroadcastJoinThreshold` | 50 MB | Auto-broadcast tables under 50 MB |