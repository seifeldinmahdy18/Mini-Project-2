"""
CSV vs Parquet Benchmark — Steam Reviews Project
=================================================

WHY PARQUET IS EXPECTED TO BE FASTER THAN CSV
-----------------------------------------------
CSV is a row-oriented text format: every field is stored as a UTF-8 string on a
single line, with no embedded type information or indexing. To read even one
column Spark must:
  1. Decompress (if gzipped) or read raw bytes for every line.
  2. Tokenise each line on the delimiter.
  3. Deserialise every field, even those that are never used.

Parquet is a columnar binary format designed for analytical workloads. Each
column is stored separately in typed, compressed chunks. This unlocks two
major optimisations that CSV cannot offer.

COLUMNAR STORAGE AND COLUMN PRUNING
--------------------------------------
Because each column occupies its own section of the file, Spark can seek
directly to the columns it needs and skip the rest entirely. A query that only
reads appid and weighted_vote_score never touches the bytes for review,
steamid, or any other column. For a wide schema this can reduce I/O by 80–90%.
CSV forces Spark to read and parse every column on every row, even if the
query references only two.

PREDICATE PUSHDOWN AT THE ROW-GROUP LEVEL
------------------------------------------
A Parquet file is divided into row-groups (typically 128 MB each). Each
row-group stores min/max statistics for every column in its footer. When a
WHERE clause filters on a numeric column (e.g. playtime_forever >= 3000),
the Parquet reader checks those statistics before decoding a single value:
  - If the row-group's max(playtime_forever) < 3000, the entire group is
    skipped — no decompression, no deserialisation.
  - Only row-groups that could contain matching rows are decoded.
CSV has no such metadata; every row must be parsed and evaluated.

WHY CSV REQUIRES FULL ROW DESERIALISATION
------------------------------------------
Even with a schema hint, Spark must parse every comma-separated field on every
line to advance to the next record. There is no way to jump from one row to the
next without reading the entire current row, because rows are variable-length
text. Field parsing is purely CPU-bound with no opportunity for vectorised I/O.

WHY THE COLUMN SCAN BENCHMARK BEST ISOLATES THE COLUMNAR ADVANTAGE
--------------------------------------------------------------------
The Column Scan query selects only three columns (appid, weighted_vote_score,
playtime_forever) out of thirteen, with no filter. There are no predicates to
push down, no aggregation to pay for — the only work is I/O. Parquet reads
~23% of the column bytes; CSV reads 100%. The speedup observed here is the
purest measure of columnar I/O savings.
"""

import os
import time

os.environ["JAVA_HOME"] = "/usr/local/sdkman/candidates/java/21.0.10-ms"
os.environ["JAVA_TOOL_OPTIONS"] = (
    "-Djava.security.manager=allow "
    "--add-opens=java.base/javax.security.auth=ALL-UNNAMED"
)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    BooleanType, LongType, IntegerType, StringType, FloatType,
)

CSV_PATH     = "data/reviews-1-115.csv"
PARQUET_PATH = "data/steam_reviews.parquet"

REVIEW_SCHEMA = StructType([
    StructField("steamid",                LongType(),    nullable=False),
    StructField("appid",                  IntegerType(), nullable=False),
    StructField("voted_up",               BooleanType(), nullable=True),
    StructField("votes_up",               IntegerType(), nullable=True),
    StructField("votes_funny",            LongType(),    nullable=True),
    StructField("weighted_vote_score",    FloatType(),   nullable=True),
    StructField("playtime_forever",       LongType(),    nullable=True),
    StructField("playtime_at_review",     LongType(),    nullable=True),
    StructField("num_games_owned",        IntegerType(), nullable=True),
    StructField("num_reviews",            IntegerType(), nullable=True),
    StructField("review",                 StringType(),  nullable=True),
    StructField("unix_timestamp_created", LongType(),    nullable=True),
    StructField("unix_timestamp_updated", LongType(),    nullable=True),
])

spark = (
    SparkSession.builder
    .appName("CSVvsParquet")
    .master("local[*]")
    .config("spark.driver.memory",                  "4g")
    .config("spark.sql.adaptive.enabled",           "false")
    .config("spark.sql.shuffle.partitions",         "200")
    .config("spark.serializer",                     "org.apache.spark.serializer.KryoSerializer")
    .config("spark.sql.parquet.filterPushdown",     "true")
    .config("spark.sql.parquet.mergeSchema",        "false")
    .config("spark.sql.autoBroadcastJoinThreshold", "52428800")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")


def run_benchmarks(df, format_label: str) -> dict:
    times = {}
    print(f"\n  [{format_label}] Running benchmarks ...")

    # Count
    t0 = time.perf_counter()
    df.count()
    times["Count"] = time.perf_counter() - t0
    print(f"    Count           : {times['Count']:.3f}s")

    # Aggregation (Q2)
    t0 = time.perf_counter()
    (
        df.groupBy("appid")
        .agg(
            F.avg("weighted_vote_score").alias("avg_helpfulness"),
            F.count("*").alias("total_reviews"),
            F.avg("playtime_forever").alias("avg_playtime"),
            F.sum("votes_up").alias("total_votes_up"),
        )
        .filter(F.col("total_reviews") >= 50)
        .count()
    )
    times["Aggregation"] = time.perf_counter() - t0
    print(f"    Aggregation     : {times['Aggregation']:.3f}s")

    # Complex Filter (Q1)
    t0 = time.perf_counter()
    (
        df
        .filter(F.col("playtime_forever")    >= 3_000)
        .filter(F.col("weighted_vote_score") >= 0.7)
        .filter(F.col("voted_up")            == True)
        .filter(F.length("review")           >= 100)
        .count()
    )
    times["Complex Filter"] = time.perf_counter() - t0
    print(f"    Complex Filter  : {times['Complex Filter']:.3f}s")

    # Column Scan
    t0 = time.perf_counter()
    df.select("appid", "weighted_vote_score", "playtime_forever").count()
    times["Column Scan"] = time.perf_counter() - t0
    print(f"    Column Scan     : {times['Column Scan']:.3f}s")

    return times


def print_comparison(csv_times: dict, parquet_times: dict) -> None:
    benchmarks = ["Count", "Aggregation", "Complex Filter", "Column Scan"]
    sep  = "=" * 62
    dash = "-" * 62

    print(f"\n{sep}")
    print(f"{'FILE FORMAT BENCHMARK: CSV vs PARQUET':^62}")
    print(sep)
    print(f"{'Benchmark':<16}| {'CSV (s)':<11}| {'Parquet (s)':<13}| Speedup")
    print(f"{'-'*16}|{'-'*12}|{'-'*14}|{'-'*8}")

    speedups = []
    for b in benchmarks:
        csv_t     = csv_times[b]
        parquet_t = parquet_times[b]
        speedup   = round(csv_t / parquet_t, 2) if parquet_t > 0 else float("inf")
        speedups.append(speedup)
        note = " *" if speedup < 1.0 else ""
        print(f"{b:<16}| {csv_t:<11.3f}| {parquet_t:<13.3f}| {speedup:.2f}x{note}")

    print(sep)

    avg_speedup = sum(speedups) / len(speedups)
    if avg_speedup >= 1.0:
        print(f"Overall: Parquet was {avg_speedup:.1f}x faster on average")
    else:
        print(f"Overall: CSV was {1/avg_speedup:.1f}x faster on average (* unexpected)")

    any_slower = [b for b, s in zip(benchmarks, speedups) if s < 1.0]
    if any_slower:
        print(f"  * Parquet was slower for: {', '.join(any_slower)}")

    print(f"{sep}\n")


if __name__ == "__main__":
    # --- CSV ---
    print("\n[INFO] Loading CSV ...")
    csv_df = (
        spark.read
        .schema(REVIEW_SCHEMA)
        .option("header",    "true")
        .option("multiLine", "true")
        .option("escape",    '"')
        .csv(CSV_PATH)
        .withColumn("voted_up",     F.col("voted_up").cast("boolean"))
        .withColumn("review_year",  F.year(F.to_timestamp("unix_timestamp_created")))
        .withColumn("review_month", F.month(F.to_timestamp("unix_timestamp_created")))
    )
    csv_times = run_benchmarks(csv_df, "CSV")

    # --- Parquet ---
    print("\n[INFO] Loading Parquet ...")
    parquet_df = (
        spark.read
        .parquet(PARQUET_PATH)
    )
    parquet_times = run_benchmarks(parquet_df, "Parquet")

    print_comparison(csv_times, parquet_times)

    spark.stop()
