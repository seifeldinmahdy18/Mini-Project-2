"""
Scalability Test — Shuffle Partition Benchmarking
===================================================

WHAT PARTITIONING MEANS IN SPARK
----------------------------------
Every Spark DataFrame is divided into partitions — independent chunks of rows
that can be processed in parallel across executor cores. When a shuffle occurs
(GROUP BY, JOIN, DISTINCT), Spark redistributes data into a new set of
partitions whose count is controlled by spark.sql.shuffle.partitions. This
setting is the single most impactful tuning knob for most workloads.

WHY AQE IS DISABLED
---------------------
Adaptive Query Execution (AQE) coalesces small post-shuffle partitions at
runtime, which means the effective partition count diverges from the configured
value. Disabling AQE (spark.sql.adaptive.enabled = false) forces Spark to
use exactly the configured spark.sql.shuffle.partitions for every shuffle,
making results directly comparable across runs.

EXPECTED PERFORMANCE TREND
----------------------------
  Too few partitions (e.g. 50):
    - Each partition holds more data → higher memory pressure per task.
    - Fewer tasks than available cores → cores sit idle, no parallelism gain.
    - Risk of spill to disk for large aggregations.

  Sweet spot (typically 100–200 for ~5 GB on 2 cores):
    - Partition size lands near the 128 MB target.
    - Tasks are plentiful enough to keep all cores busy without scheduler
      overhead dominating.

  Too many partitions (e.g. 400+):
    - Hundreds of tiny tasks → task-launch overhead can exceed compute time.
    - Network overhead increases: more small shuffle files to transfer/merge.
    - Scheduler spends more time assigning work than executing it.

HOW SHUFFLE PARTITIONS RELATE TO TASK COUNT
---------------------------------------------
Each shuffle partition becomes exactly one reduce-side task. So
spark.sql.shuffle.partitions = 200 means Spark creates 200 tasks after every
shuffle boundary. With local[*] and N cores, at most N tasks run in parallel;
the remaining tasks queue. The goal is to have enough tasks that all cores stay
busy, but not so many that the overhead of managing thousands of empty tasks
cancels out the parallelism benefit.
"""

import os
import time

os.environ["JAVA_HOME"] = "/usr/local/sdkman/candidates/java/21.0.10-ms"
os.environ["JAVA_TOOL_OPTIONS"] = (
    "-Djava.security.manager=allow "
    "--add-opens=java.base/javax.security.auth=ALL-UNNAMED"
)

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    BooleanType, LongType, IntegerType, StringType, FloatType,
)

DATA_PATH = "data/steam_reviews.parquet"

PARTITION_COUNTS = [50, 100, 200, 400]

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


def build_session(partitions: int) -> SparkSession:
    return (
        SparkSession.builder
        .appName(f"ScalabilityTest-p{partitions}")
        .master("local[*]")
        .config("spark.driver.memory",                   "4g")
        .config("spark.sql.shuffle.partitions",          str(partitions))
        .config("spark.sql.adaptive.enabled",            "false")
        .config("spark.serializer",                      "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.parquet.filterPushdown",      "true")
        .config("spark.sql.parquet.mergeSchema",         "false")
        .config("spark.sql.autoBroadcastJoinThreshold",  "52428800")
        .getOrCreate()
    )


def run_q2(reviews) -> float:
    t0 = time.perf_counter()
    (
        reviews
        .groupBy("appid")
        .agg(
            F.avg("weighted_vote_score").alias("avg_helpfulness"),
            F.count("*").alias("total_reviews"),
            F.avg("playtime_forever").alias("avg_playtime"),
            F.sum("votes_up").alias("total_votes_up"),
        )
        .filter(F.col("total_reviews") >= 50)
        .count()
    )
    return time.perf_counter() - t0


def run_q3(reviews) -> float:
    t0 = time.perf_counter()
    (
        reviews
        .groupBy("appid", "voted_up")
        .agg(
            F.count("*").alias("review_count"),
            F.avg("weighted_vote_score").alias("avg_helpfulness"),
        )
        .count()
    )
    return time.perf_counter() - t0


def run_q5(reviews) -> float:
    influence_window = (
        Window
        .partitionBy("appid")
        .orderBy(F.col("weighted_vote_score").desc())
    )
    t0 = time.perf_counter()
    (
        reviews
        .filter(F.col("weighted_vote_score").isNotNull())
        .withColumn("influence_rank", F.rank().over(influence_window))
        .filter(F.col("influence_rank") == 1)
        .count()
    )
    return time.perf_counter() - t0


def print_results(results: list[dict]) -> None:
    sep  = "=" * 62
    dash = "-" * 62

    print(f"\n{sep}")
    print(f"{'SCALABILITY TEST RESULTS':^62}")
    print(sep)
    print(f"{'Partitions':<12}| {'Q2 Agg (s)':<14}| {'Q3 Group (s)':<16}| {'Q5 Window (s)'}")
    print(dash)
    for r in results:
        print(
            f"{r['partitions']:<12}| "
            f"{r['q2']:<14.3f}| "
            f"{r['q3']:<16.3f}| "
            f"{r['q5']:.3f}"
        )
    print(sep)

    best_q2 = min(results, key=lambda r: r["q2"])
    best_q3 = min(results, key=lambda r: r["q3"])
    best_q5 = min(results, key=lambda r: r["q5"])

    print("Best partition count per query:")
    print(f"  Q2 Agg:    {best_q2['partitions']:>3} partitions ({best_q2['q2']:.3f}s)")
    print(f"  Q3 Group:  {best_q3['partitions']:>3} partitions ({best_q3['q3']:.3f}s)")
    print(f"  Q5 Window: {best_q5['partitions']:>3} partitions ({best_q5['q5']:.3f}s)")
    print(f"{sep}\n")


if __name__ == "__main__":
    results = []

    for partitions in PARTITION_COUNTS:
        print(f"\n[INFO] Starting session with {partitions} shuffle partitions ...")
        spark = build_session(partitions)
        spark.sparkContext.setLogLevel("WARN")

        reviews = (
            spark.read
            .schema(REVIEW_SCHEMA)
            .parquet(DATA_PATH)
        )

        print(f"  Running Q2 (Aggregation) ...")
        t_q2 = run_q2(reviews)
        print(f"  Q2 done: {t_q2:.3f}s")

        print(f"  Running Q3 (Multi-group) ...")
        t_q3 = run_q3(reviews)
        print(f"  Q3 done: {t_q3:.3f}s")

        print(f"  Running Q5 (Window Rank) ...")
        t_q5 = run_q5(reviews)
        print(f"  Q5 done: {t_q5:.3f}s")

        results.append({"partitions": partitions, "q2": t_q2, "q3": t_q3, "q5": t_q5})

        spark.stop()
        print(f"[INFO] Session stopped for {partitions} partitions.")

    print_results(results)
