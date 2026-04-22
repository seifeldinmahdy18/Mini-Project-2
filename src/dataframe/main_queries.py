# main_queries.py
# PySpark DataFrame API — 10 queries on 15.4M Steam Reviews
# Dataset: kaggle.com/datasets/forgemaster/steam-reviews-dataset

import time
from contextlib import contextmanager

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import (
    StructType, StructField,
    LongType, IntegerType, StringType, BooleanType, FloatType,
)


# =============================================================================
# SPARK SESSION
# AQE lets Spark re-optimize the plan at runtime (e.g. auto-upgrade a
# shuffle join to broadcast if a partition turns out small).
# Kryo serializer is ~5x faster than Java default for inter-executor data.
# =============================================================================

spark = (
    SparkSession.builder
    .appName("SteamReviews_DataFrame_Queries")
    .config("spark.sql.adaptive.enabled",                    "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.sql.adaptive.skewJoin.enabled",           "true")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.kryoserializer.buffer.max",               "512m")
    # Target ~128 MB per partition. 200 is right for 5 GB; scale up for larger data.
    .config("spark.sql.shuffle.partitions",                  "200")
    # Push filters into the Parquet reader so unneeded row-groups are skipped.
    .config("spark.sql.parquet.filterPushdown",              "true")
    .config("spark.sql.parquet.mergeSchema",                 "false")
    .config("spark.sql.autoBroadcastJoinThreshold",          str(50 * 1024 * 1024))
    .config("spark.memory.fraction",                         "0.8")
    .config("spark.memory.storageFraction",                  "0.3")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# =============================================================================
# SCHEMAS
# Defining schemas explicitly avoids Spark scanning the full dataset just
# to infer types — an expensive no-op before any real work starts.
# =============================================================================

REVIEW_SCHEMA = StructType([
    StructField("review_id",               LongType(),    nullable=False),
    StructField("app_id",                  LongType(),    nullable=False),
    StructField("author_steamid",          LongType(),    nullable=True),
    StructField("language",                StringType(),  nullable=True),
    StructField("review",                  StringType(),  nullable=True),
    StructField("timestamp_created",       LongType(),    nullable=True),   # Unix epoch
    StructField("recommended",             BooleanType(), nullable=True),
    StructField("votes_helpful",           IntegerType(), nullable=True),
    StructField("votes_funny",             IntegerType(), nullable=True),
    StructField("weighted_vote_score",     FloatType(),   nullable=True),
    StructField("author_playtime_forever", IntegerType(), nullable=True),   # minutes
])

APP_META_SCHEMA = StructType([
    StructField("app_id",   LongType(),   nullable=False),
    StructField("app_name", StringType(), nullable=True),
    StructField("genre",    StringType(), nullable=True),
])

USER_PROFILE_SCHEMA = StructType([
    StructField("author_steamid",   LongType(),    nullable=False),
    StructField("username",         StringType(),  nullable=True),
    StructField("account_age_days", IntegerType(), nullable=True),
    StructField("total_reviews",    IntegerType(), nullable=True),
])


# =============================================================================
# DATA LOADING
# Swap DATA_PATH for your GCS / S3 / HDFS URI in the cloud environment.
# For CSV: spark.read.schema(REVIEW_SCHEMA).option("header", True).csv(path)
# review_year / review_month are derived once here and reused by Q6 and Q9.
# =============================================================================

DATA_PATH = "data/steam_reviews.parquet"

reviews = (
    spark.read
    .schema(REVIEW_SCHEMA)
    .parquet(DATA_PATH)
    .withColumn("review_year",  F.year(F.to_timestamp("timestamp_created")))
    .withColumn("review_month", F.month(F.to_timestamp("timestamp_created")))
)


# =============================================================================
# TIMER UTILITY
# Wraps any action (count / show) and prints wall-clock time.
# =============================================================================

@contextmanager
def timer(label: str):
    print(f"\n[START] {label}")
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"[DONE]  {label} — {time.perf_counter() - start:.3f}s\n")


# =============================================================================
# QUERY 1 — COMPLEX FILTER
# Goal: Isolate high-quality "Elite" reviews using four conditions.
#
# Catalyst behaviour: all four filter predicates are fused into a single
# compound node in the Optimized Logical Plan. Numeric predicates are
# pushed down to the Parquet row-group level, so entire file chunks are
# skipped before data enters JVM memory.
# Expected result: ~5% of 15.4M rows survive (roughly 600K–900K).
# =============================================================================

elite_df = (
    reviews
    .filter(F.col("author_playtime_forever") >= 3_000)  # 50+ hours played
    .filter(F.col("weighted_vote_score")     >= 0.7)    # peer-validated helpfulness
    .filter(F.col("recommended")             == True)
    .filter(F.length("review")               >= 100)    # non-trivial written review
    .select(
        "review_id", "app_id", "author_steamid", "language", "review",
        "timestamp_created", "votes_helpful", "weighted_vote_score",
        "author_playtime_forever", "review_year",
    )
)

print("\n--- Q1 Execution Plan (Complex Filter) ---")
elite_df.explain(True)
# In the Physical Plan look for:
#   PushedFilters: [IsNotNull(...), GreaterThanOrEqual(...), ...]
#   This confirms Parquet row-group skipping is active.

# Cache because Q7 (subquery) reuses this DataFrame.
elite_df.cache()

with timer("Q1 — Complex Filter"):
    print(f"  Elite reviewers: {elite_df.count():,}")


# =============================================================================
# QUERY 2 — AGGREGATION
# Goal: Compute average helpfulness, review count, and total helpful votes
#       per game (only games with >= 50 reviews are included).
#
# Catalyst behaviour: groupBy + agg triggers a partial Hash Aggregate on
# each executor before the shuffle. Only pre-aggregated summaries cross
# the network, reducing shuffle bytes by 80–90%.
# =============================================================================

per_game_stats_df = (
    reviews
    .groupBy("app_id")
    .agg(
        F.round(F.avg("weighted_vote_score"), 4).alias("avg_helpfulness"),
        F.count("review_id").alias("total_reviews"),
        F.round(F.avg("author_playtime_forever"), 2).alias("avg_playtime_mins"),
        F.sum("votes_helpful").alias("total_votes_helpful"),
    )
    .filter(F.col("total_reviews") >= 50)
)

with timer("Q2 — Aggregation (avg helpfulness per game)"):
    per_game_stats_df.show(10, truncate=False)


# =============================================================================
# QUERY 3 — MULTI-ATTRIBUTE GROUPING
# Goal: Group by language AND recommended to see how review sentiment and
#       helpfulness differ across language communities.
#
# Catalyst behaviour: multi-key groupBy produces a wider shuffle key.
# AQE coalesces small post-shuffle partitions automatically, avoiding
# the overhead of many near-empty tasks.
# =============================================================================

by_language_sentiment_df = (
    reviews
    .filter(F.col("language").isNotNull())
    .filter(F.col("recommended").isNotNull())
    .groupBy("language", "recommended")
    .agg(
        F.count("review_id").alias("review_count"),
        F.round(F.avg("weighted_vote_score"), 4).alias("avg_helpfulness"),
        F.round(F.avg("author_playtime_forever"), 2).alias("avg_playtime_mins"),
        F.round(F.stddev("weighted_vote_score"), 4).alias("score_std_dev"),
    )
    .orderBy(F.col("review_count").desc())
)

with timer("Q3 — Multi-Attribute Grouping (language x recommendation)"):
    by_language_sentiment_df.show(20, truncate=False)


# =============================================================================
# QUERY 4 — SORTING & RANKING
# Goal: Find the top 10 games by a composite engagement score:
#         engagement = total_reviews * avg_helpfulness * log(avg_playtime + 1)
#
# Catalyst behaviour: ORDER BY + LIMIT is rewritten as TakeOrderedAndProject.
# Each executor maintains a local top-K heap, so only K rows per executor
# travel to the driver — far cheaper than a full global sort.
# =============================================================================

top10_games_df = (
    per_game_stats_df
    .withColumn(
        "engagement_score",
        F.round(
            F.col("total_reviews") * F.col("avg_helpfulness") * F.log1p("avg_playtime_mins"),
            2,
        )
    )
    .orderBy(F.col("engagement_score").desc())
    .limit(10)
)

with timer("Q4 — Sorting & Ranking (top 10 games by engagement)"):
    top10_games_df.show(10, truncate=False)


# =============================================================================
# QUERY 5 — WINDOW FUNCTION: RANKING
# Goal: Rank reviews within each game by helpfulness score and keep only
#       the single most influential review per app_id.
#
# Catalyst behaviour: generates an Exchange hashpartitioning(app_id) shuffle
# so all reviews for the same game share an executor. With AQE enabled,
# skewed partitions (games with 1M+ reviews) are split across multiple tasks.
# =============================================================================

influence_window = (
    Window
    .partitionBy("app_id")
    .orderBy(F.col("weighted_vote_score").desc(), F.col("votes_helpful").desc())
)

top_review_per_game_df = (
    reviews
    .filter(F.col("weighted_vote_score").isNotNull())
    .withColumn("influence_rank", F.rank().over(influence_window))
    .filter(F.col("influence_rank") == 1)
    .select(
        "app_id", "review_id", "author_steamid", "language",
        "recommended", "weighted_vote_score", "votes_helpful",
        "author_playtime_forever", "influence_rank",
    )
)

print("\n--- Q5 Execution Plan (Window Rank) ---")
top_review_per_game_df.explain(True)
# Look for: Window [rank() OVER (PARTITION BY app_id ORDER BY ...)]
#           Exchange hashpartitioning(app_id, 200) — the unavoidable shuffle.

with timer("Q5 — Window Rank (top review per game)"):
    print(f"  Games with a ranked top review: {top_review_per_game_df.count():,}")


# =============================================================================
# QUERY 6 — WINDOW FUNCTION: MOVING AVERAGE
# Goal: Compute a 7-day rolling average of daily review volume per game
#       to surface trend momentum over time.
#
# Note: pre-aggregate to daily counts BEFORE applying the window.
# Running a range window directly on 15.4M raw rows causes large memory spill.
# =============================================================================

daily_counts_df = (
    reviews
    .withColumn("review_date", F.to_date(F.to_timestamp("timestamp_created")))
    .filter(F.col("review_date").isNotNull())
    .groupBy("app_id", "review_date")
    .agg(F.count("review_id").alias("daily_review_count"))
)

# rangeBetween uses epoch-seconds because the order column is cast to long.
hype_window = (
    Window
    .partitionBy("app_id")
    .orderBy(F.col("review_date").cast("long"))
    .rangeBetween(-(6 * 86_400), 0)   # current day + 6 prior days
)

hype_trend_df = (
    daily_counts_df
    .withColumn("avg_7d_reviews", F.round(F.avg("daily_review_count").over(hype_window), 2))
    .orderBy("app_id", "review_date")
)

with timer("Q6 — Moving Average (7-day review trend)"):
    hype_trend_df.show(15, truncate=False)


# =============================================================================
# QUERY 7 — NESTED / SUBQUERY LOGIC
# Goal: Return elite reviews that belong to games whose average helpfulness
#       score exceeds the global average (a two-phase filter).
#
# Catalyst behaviour: the global average is computed as a scalar subquery in
# a separate scan pass, then broadcast to the outer filter — the dataset is
# not read twice. Look for a "Subquery" branch in the Physical Plan.
# =============================================================================

# Phase 1: per-game average (min 20 reviews for statistical validity)
per_game_avg_df = (
    reviews
    .filter(F.col("weighted_vote_score").isNotNull())
    .groupBy("app_id")
    .agg(
        F.round(F.avg("weighted_vote_score"), 4).alias("game_avg_score"),
        F.count("review_id").alias("game_review_count"),
    )
    .filter(F.col("game_review_count") >= 20)
)

# Phase 2: compute global average as a Python scalar
global_avg: float = (
    reviews
    .filter(F.col("weighted_vote_score").isNotNull())
    .agg(F.avg("weighted_vote_score"))
    .collect()[0][0]
)
print(f"\n  Global average helpfulness score: {global_avg:.4f}")

# Phase 3: filter above-average games, then join back to elite reviewers
above_avg_elite_df = (
    elite_df
    .join(
        per_game_avg_df.filter(F.col("game_avg_score") > global_avg),
        on="app_id",
        how="inner",
    )
    .select(
        "review_id", "app_id", "author_steamid", "language",
        "weighted_vote_score", "game_avg_score", "author_playtime_forever",
    )
)

with timer("Q7 — Subquery (elite reviews in above-average games)"):
    print(f"  Matching reviews: {above_avg_elite_df.count():,}")
    print(f"  Global avg threshold: {global_avg:.4f}")


# =============================================================================
# QUERY 8 — BROADCAST JOIN
# Goal: Enrich the 15.4M reviews with game names and genres from a small
#       app_metadata lookup table without triggering a shuffle.
#
# Why broadcast: the metadata table is tiny (< 1 MB). F.broadcast() tells
# Catalyst to copy it to every executor's memory as a hash table. Each
# executor performs a local lookup — zero bytes cross the network for
# the metadata side. Compare to a Sort-Merge Join where both sides shuffle.
# Only use broadcast for tables reliably under ~200 MB.
# =============================================================================

app_metadata_df = spark.createDataFrame(
    [
        (440,     "Team Fortress 2",    "Action"),
        (570,     "Dota 2",             "Strategy"),
        (730,     "CS:GO",              "FPS"),
        (578080,  "PUBG",               "Battle Royale"),
        (1172620, "Sea of Thieves",     "Adventure"),
        (1091500, "Cyberpunk 2077",     "RPG"),
        (945360,  "Among Us",           "Social Deduction"),
        (1422450, "Vampire Survivors",  "Rogue-like"),
    ],
    schema=APP_META_SCHEMA,
)

broadcast_joined_df = (
    reviews
    .join(F.broadcast(app_metadata_df), on="app_id", how="inner")
    .select("app_id", "app_name", "genre", "review_id",
            "recommended", "weighted_vote_score", "author_playtime_forever")
)

print("\n--- Q8 Execution Plan (Broadcast Join) ---")
broadcast_joined_df.explain(True)
# Look for: BroadcastHashJoin (not SortMergeJoin) and BroadcastExchange
# on the metadata side. No Exchange hashpartitioning for app_metadata.

with timer("Q8 — Broadcast Join (reviews x app metadata)"):
    broadcast_joined_df.show(10, truncate=False)


# =============================================================================
# QUERY 9 — PARTITION PRUNING
# Goal: Analyse monthly review volumes and average scores for 2022–2023 only,
#       demonstrating how partitioned storage eliminates unnecessary I/O.
#
# How it works: when data is written with .write.partitionBy("review_year",
# "review_month"), Spark organises files into subdirectories by year/month.
# Filtering on review_year allows Spark to open only those directories —
# all other years are never read from disk.
#
# To enable real pruning, write once (uncomment below), then re-read with
# spark.read.parquet(PARTITIONED_PATH) and apply the year filter.
# =============================================================================

PARTITIONED_PATH = "data/steam_reviews_partitioned.parquet"

# Write partitioned output (run once; produces ~5 GB, so skipped by default):
# reviews.write.partitionBy("review_year", "review_month") \
#         .mode("overwrite").parquet(PARTITIONED_PATH)

pruned_df = (
    reviews
    .filter(F.col("review_year").isin(2022, 2023))
    .groupBy("review_year", "review_month")
    .agg(
        F.count("review_id").alias("monthly_reviews"),
        F.round(F.avg("weighted_vote_score"), 4).alias("avg_score"),
    )
    .orderBy("review_year", "review_month")
)

print("\n--- Q9 Execution Plan (Partition Pruning) ---")
pruned_df.explain(True)
# After writing with partitionBy and re-reading, the Physical Plan shows:
#   PartitionFilters: [review_year IN (2022, 2023)]
# This means only those subdirectories are opened — all other years are skipped.

with timer("Q9 — Partition Pruning (2022–2023 monthly summary)"):
    pruned_df.show(24, truncate=False)


# =============================================================================
# QUERY 10 — SORT-MERGE JOIN
# Goal: Join the 15.4M reviews with a large user_profiles table on
#       author_steamid to add account-level context.
#
# Why Sort-Merge Join: neither side is small enough to broadcast.
# Spark sorts both tables by the join key, then walks them in lockstep
# (like a zipper), matching rows without loading either table entirely.
# This requires two Exchange (shuffle) nodes — one per side — but it is
# the only scalable strategy for large × large joins.
#
# Optimisation: pre-calling .repartition(200, "author_steamid") before
# the join aligns the partitioning scheme upfront. Spark detects this and
# can skip the re-partition Exchange on that side, saving one shuffle pass.
# =============================================================================

user_profiles_df = spark.createDataFrame(
    [
        (76561198000000001, "NightStrike",  1200, 315),
        (76561198000000002, "VoidWalker",   890,  42),
        (76561198000000003, "CatalystKing", 2100, 178),
        (76561198000000004, "LagSlayer",    450,  9),
        (76561198000000005, "FrostByte",    3300, 501),
        (76561198000000006, "GankMaster",   760,  88),
        (76561198000000007, "SparkArcher",  1980, 234),
        (76561198000000008, "ManaVault",    550,  61),
    ],
    schema=USER_PROFILE_SCHEMA,
)

# Pre-repartition on the join key to reduce shuffle work at join time.
reviews_pre_partitioned = reviews.repartition(200, "author_steamid")

sort_merge_joined_df = (
    reviews_pre_partitioned
    .join(user_profiles_df, on="author_steamid", how="inner")
    .select(
        "author_steamid", "username", "account_age_days",
        "total_reviews", "app_id", "review_id",
        "recommended", "weighted_vote_score", "author_playtime_forever",
    )
    .withColumn("is_veteran_account", F.col("account_age_days") >= 365)
)

print("\n--- Q10 Execution Plan (Sort-Merge Join) ---")
sort_merge_joined_df.explain(True)
# Look for: SortMergeJoin [author_steamid], [author_steamid], Inner
# Two Exchange hashpartitioning nodes (one per table).
# With pre-repartition: AQE may replace one Exchange with AQEShuffleRead.

with timer("Q10 — Sort-Merge Join (reviews x user profiles)"):
    sort_merge_joined_df.show(15, truncate=False)


# =============================================================================
# CLEANUP
# =============================================================================

elite_df.unpersist()

# Uncomment when running as a standalone script (not a cloud notebook):
# spark.stop()
