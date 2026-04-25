# Mini-Project-2 — Steam Reviews Big Data Analysis

A PySpark-based analysis of **15.4 million Steam reviews (~5 GB)** implemented across three APIs: DataFrame, RDD, and Spark SQL.

**Dataset:** [Steam Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/forgemaster/steam-reviews-dataset)

---

## Dataset

### Overview

| Property | Value |
|---|---|
| Records | 15,400,000+ |
| Raw Size | ~5 GB |
| Format | CSV → Parquet |
| Source | Kaggle — `forgemaster/steam-reviews-dataset` |

### Schema

| Column | Type | Description |
|---|---|---|
| `steamid` | Long | Steam ID of the reviewer |
| `appid` | Integer | Steam application ID of the reviewed game |
| `voted_up` | Boolean | Whether the reviewer recommended the game (`true` / `false`) |
| `votes_up` | Integer | Number of users who marked the review as helpful |
| `votes_funny` | Long | Number of users who marked the review as funny |
| `weighted_vote_score` | Float | Steam's internal helpfulness score (0.0 – 1.0) |
| `playtime_forever` | Long | Total minutes the reviewer has played the game (all time) |
| `playtime_at_review` | Long | Minutes played at the time the review was written |
| `num_games_owned` | Integer | Number of games owned by the reviewer |
| `num_reviews` | Integer | Total number of reviews written by the reviewer |
| `review` | String | Full text of the review |
| `unix_timestamp_created` | Long | Unix epoch timestamp when the review was submitted |
| `unix_timestamp_updated` | Long | Unix epoch timestamp when the review was last updated |
| `review_year` | Integer | Derived from `unix_timestamp_created` |
| `review_month` | Integer | Derived from `unix_timestamp_created` |

---

## Project Structure

```
Mini-Project-2/
├── data/                          # Raw CSV and converted Parquet dataset
├── logs/
│   ├── dataframe_output.txt       # Output log from DataFrame API run
│   ├── rdd_output.txt             # Output log from RDD API run
│   └── sql_output.txt             # Output log from Spark SQL run
├── scripts/
│   ├── fetch_data.py              # Downloads the dataset from Kaggle
│   └── convert_to_parquet.py      # Converts CSV to typed Parquet
├── src/
│   ├── dataframe/
│   │   └── main_queries.py        # 10 queries using the DataFrame API
│   ├── rdd/
│   │   └── main_queries.py        # 10 queries using the RDD API
│   └── sql/
│       └── main_queries.py        # 10 queries using Spark SQL
├── .env                           # KAGGLE_USERNAME and KAGGLE_KEY
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

### 3. Convert CSV to Parquet

```bash
python scripts/convert_to_parquet.py
```

### 4. Run the queries

```bash
# DataFrame API
python src/dataframe/main_queries.py

# RDD API
python src/rdd/main_queries.py

# Spark SQL
python src/sql/main_queries.py
```

Output logs are saved under `logs/`.

---

## The 10 Queries

Each query is implemented in all three APIs (`dataframe/`, `rdd/`, `sql/`).

| # | Query Type | Description | Key Optimization |
|---|---|---|---|
| 1 | **Complex Filter** | Filter reviews by playtime (≥ 3,000 min), helpfulness score (≥ 0.7), positive recommendation, and review length (≥ 100 chars) to isolate high-quality "Elite" reviews | Catalyst pushes all predicates to the Parquet row-group reader — unneeded file chunks are never loaded |
| 2 | **Aggregation** | Compute average helpfulness, total review count, average playtime, and total helpful votes per game (minimum 50 reviews) | Partial Hash Aggregate runs on each executor before the shuffle, reducing network bytes by 80–90% |
| 3 | **Multi-Attribute Grouping** | Group by `appid × voted_up` to compare review sentiment and helpfulness across games | AQE coalesces small post-shuffle partitions automatically, avoiding empty-task overhead |
| 4 | **Sorting & Ranking** | Rank games by a composite engagement score (`total_reviews × avg_helpfulness × log(avg_playtime + 1)`) and return the top 10 | Catalyst rewrites ORDER BY + LIMIT as `TakeOrderedAndProject`, using a per-executor top-K heap instead of a global sort |
| 5 | **Window Function — Rank** | Assign a rank to every review within its game partition (ordered by helpfulness) and keep only the single most influential review per game | Partitioned window triggers one `Exchange hashpartitioning(appid)` shuffle; AQE handles skewed partitions |
| 6 | **Window Function — Moving Average** | Compute a 7-day rolling average of daily review volume per game to surface review trend momentum | Pre-aggregate to daily counts before applying the range window; avoids memory spill from windowing 15.4M raw rows |
| 7 | **Nested / Subquery Logic** | Return elite reviews that belong to games whose average helpfulness exceeds the global average (two-phase filter) | Global average is computed as a scalar subquery in one scan pass and broadcast to the outer filter; no double scan |
| 8 | **Broadcast Join** | Join reviews with a small `app_metadata` lookup table (game name, genre) using an explicit broadcast hint | `F.broadcast()` / `/*+ BROADCAST */` sends the small table to every executor as a hash table — zero shuffle bytes on the metadata side |
| 9 | **Partition Pruning** | Filter reviews to 2019 only and aggregate monthly statistics, demonstrating directory-level I/O elimination | When data is written with `.partitionBy("review_year", "review_month")`, Spark opens only the matching subdirectories |
| 10 | **Sort-Merge Join** | Join reviews with a `user_profiles` table on `steamid` to add account-level context | Pre-repartitioning on the join key before the join aligns partitioning so Spark can skip one Exchange phase |

Each query includes:
- A `.explain(True)` call (for optimization-heavy queries) showing the full Catalyst plan
- A `timer()` wrapper that prints wall-clock execution time for every action

---

## Configuration Notes

All three `main_queries.py` files share the same `SparkSession` configuration:

| Setting | Value | Reason |
|---|---|---|
| `spark.sql.adaptive.enabled` | `true` | Runtime re-optimization (AQE) |
| `spark.sql.shuffle.partitions` | `200` | ~128 MB per partition for 5 GB; scale up for larger data |
| `spark.serializer` | Kryo | ~5× faster than Java default for inter-executor transfer |
| `spark.sql.parquet.filterPushdown` | `true` | Pushes WHERE clauses into the Parquet reader |
| `spark.sql.autoBroadcastJoinThreshold` | 50 MB | Auto-broadcast tables under 50 MB |

### Java Compatibility

PySpark requires Java 21. The scripts set `JAVA_HOME` and `JAVA_TOOL_OPTIONS` automatically:

```python
os.environ["JAVA_HOME"] = "/usr/local/sdkman/candidates/java/21.0.10-ms"
os.environ["JAVA_TOOL_OPTIONS"] = (
    "-Djava.security.manager=allow "
    "--add-opens=java.base/javax.security.auth=ALL-UNNAMED"
)
```
