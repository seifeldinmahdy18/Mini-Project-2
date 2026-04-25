# main_queries.py
# PySpark DataFrame API — 10 queries on Steam Reviews
# Dataset: kaggle.com/datasets/forgemaster/steam-reviews-dataset

import os
os.environ["JAVA_HOME"] = "/usr/local/sdkman/candidates/java/21.0.10-ms"
os.environ["JAVA_TOOL_OPTIONS"] = (
    "-Djava.security.manager=allow "
    "--add-opens=java.base/javax.security.auth=ALL-UNNAMED"
)
import time
from contextlib import contextmanager

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import (
    StructType, StructField,
    BooleanType, LongType, IntegerType, StringType, FloatType,
)


# =============================================================================
# SPARK SESSION
# AQE lets Spark re-optimize the plan at runtime (e.g. auto-upgrade a
# shuffle join to broadcast if a partition turns out small).
# Kryo serializer is ~5x faster than Java default for inter-executor data.
# =============================================================================

spark = (
    SparkSession.builder
    .appName("SteamReviews")

    .config("spark.sql.adaptive.enabled",                    "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.sql.adaptive.skewJoin.enabled",           "true")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.kryoserializer.buffer.max",               "512m")
    .config("spark.sql.shuffle.partitions",                  "200")
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

APP_META_SCHEMA = StructType([
    StructField("appid",    IntegerType(), nullable=False),
    StructField("app_name", StringType(),  nullable=True),
    StructField("genre",    StringType(),  nullable=True),
])

USER_PROFILE_SCHEMA = StructType([
    StructField("steamid",          LongType(),    nullable=False),
    StructField("username",         StringType(),  nullable=True),
    StructField("account_age_days", IntegerType(), nullable=True),
    StructField("total_reviews",    IntegerType(), nullable=True),
])


# =============================================================================
# DATA LOADING
# review_year / review_month are derived once here and reused by Q6 and Q9.
# =============================================================================

DATA_PATH = "data/steam_reviews.parquet"

reviews = (
    spark.read
    .schema(REVIEW_SCHEMA)
    .parquet(DATA_PATH)
    .withColumn("review_year",  F.year(F.to_timestamp("unix_timestamp_created")))
    .withColumn("review_month", F.month(F.to_timestamp("unix_timestamp_created")))
)


# =============================================================================
# LOOKUP TABLES
# =============================================================================

app_metadata_df = spark.createDataFrame([
    (10,      "Counter-Strike",    "Action"),
    (570,     "Dota 2",            "Strategy"),
    (730,     "CS:GO",             "FPS"),
    (578080,  "PUBG",              "Battle Royale"),
    (1172620, "Sea of Thieves",    "Adventure"),
    (1091500, "Cyberpunk 2077",    "RPG"),
    (945360,  "Among Us",          "Social Deduction"),
    (1422450, "Vampire Survivors", "Rogue-like"),
], schema=APP_META_SCHEMA)

user_profiles_df = spark.createDataFrame([
    (76561198107294407, "NightStrike",  1200, 315),
    (76561198011733201, "VoidWalker",   890,  42),
    (76561198000000003, "CatalystKing", 2100, 178),
    (76561198000000004, "LagSlayer",    450,  9),
    (76561198000000005, "FrostByte",    3300, 501),
    (76561198000000006, "GankMaster",   760,  88),
    (76561198000000007, "SparkArcher",  1980, 234),
    (76561198000000008, "ManaVault",    550,  61),
], schema=USER_PROFILE_SCHEMA)


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
# pushed down to the Parquet row-group level so entire file chunks are
# skipped before data enters JVM memory.
# =============================================================================

elite_df = (
    reviews
    .filter(F.col("playtime_forever")    >= 3_000)
    .filter(F.col("weighted_vote_score") >= 0.7)
    .filter(F.col("voted_up")            == True)
    .filter(F.length("review")           >= 100)
    .select(
        "steamid", "appid", "voted_up", "votes_up",
        "weighted_vote_score", "playtime_forever",
        "review", "unix_timestamp_created", "review_year",
    )
)

print("\n--- Q1 Execution Plan (Complex Filter) ---")
elite_df.explain(True)
# In the Physical Plan look for:
#   PushedFilters: [IsNotNull(...), GreaterThanOrEqual(...), ...]

# Cache because Q7 reuses this DataFrame.
elite_df.cache()

with timer("Q1 — Complex Filter"):
    print(f"  Elite reviewers: {elite_df.count():,}")


# =============================================================================
# QUERY 2 — AGGREGATION
# Goal: Compute average helpfulness, review count, avg playtime, and total
#       votes_up per game (only games with >= 50 reviews).
#
# Catalyst behaviour: groupBy + agg triggers a partial Hash Aggregate on
# each executor before the shuffle. Only pre-aggregated summaries cross
# the network, reducing shuffle bytes by 80–90%.
# =============================================================================

per_game_stats_df = (
    reviews
    .groupBy("appid")
    .agg(
        F.round(F.avg("weighted_vote_score"), 4).alias("avg_helpfulness"),
        F.count("*").alias("total_reviews"),
        F.round(F.avg("playtime_forever"), 2).alias("avg_playtime_mins"),
        F.sum("votes_up").alias("total_votes_up"),
    )
    .filter(F.col("total_reviews") >= 50)
)

with timer("Q2 — Aggregation (avg helpfulness per game)"):
    per_game_stats_df.show(10, truncate=False)


# =============================================================================
# QUERY 3 — MULTI-ATTRIBUTE GROUPING
# Goal: Group by appid AND voted_up to see how review sentiment and
#       helpfulness differ across games.
#
# Catalyst behaviour: multi-key groupBy produces a wider shuffle key.
# AQE coalesces small post-shuffle partitions automatically, avoiding
# the overhead of many near-empty tasks.
# =============================================================================

by_app_sentiment_df = (
    reviews
    .filter(F.col("voted_up").isNotNull())
    .filter(F.col("appid").isNotNull())
    .groupBy("appid", "voted_up")
    .agg(
        F.count("*").alias("review_count"),
        F.round(F.avg("weighted_vote_score"), 4).alias("avg_helpfulness"),
        F.round(F.avg("playtime_forever"), 2).alias("avg_playtime_mins"),
        F.round(F.stddev("weighted_vote_score"), 4).alias("score_std_dev"),
    )
    .orderBy(F.col("review_count").desc())
)

with timer("Q3 — Multi-Attribute Grouping (appid x voted_up)"):
    by_app_sentiment_df.show(20, truncate=False)


# =============================================================================
# QUERY 4 — SORTING & RANKING
# Goal: Find the top 10 games by a composite engagement score:
#         engagement = total_reviews * avg_helpfulness * log(avg_playtime + 1)
#
# Catalyst behaviour: ORDER BY + LIMIT is rewritten as TakeOrderedAndProject.
# Each executor maintains a local top-K heap so only K rows per executor
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
# Goal: Rank reviews within each game by weighted_vote_score and keep only
#       the single most influential review per appid.
#
# Catalyst behaviour: generates an Exchange hashpartitioning(appid) shuffle
# so all reviews for the same game share an executor. With AQE enabled,
# skewed partitions are split across multiple tasks.
# =============================================================================

influence_window = (
    Window
    .partitionBy("appid")
    .orderBy(F.col("weighted_vote_score").desc(), F.col("votes_up").desc())
)

top_review_per_game_df = (
    reviews
    .filter(F.col("weighted_vote_score").isNotNull())
    .withColumn("influence_rank", F.rank().over(influence_window))
    .filter(F.col("influence_rank") == 1)
    .select(
        "appid", "steamid", "voted_up", "weighted_vote_score",
        "votes_up", "playtime_forever", "influence_rank",
    )
)

print("\n--- Q5 Execution Plan (Window Rank) ---")
top_review_per_game_df.explain(True)
# Look for: Window [rank() OVER (PARTITION BY appid ORDER BY ...)]
#           Exchange hashpartitioning(appid, 200)

with timer("Q5 — Window Rank (top review per game)"):
    print(f"  Games with a ranked top review: {top_review_per_game_df.count():,}")


# =============================================================================
# QUERY 6 — WINDOW FUNCTION: MOVING AVERAGE
# Goal: Compute a 7-day rolling average of daily review volume per game
#       to surface trend momentum over time.
#
# Note: pre-aggregate to daily counts BEFORE applying the window.
# Running a range window directly on raw rows causes large memory spill.
# =============================================================================

daily_counts_df = (
    reviews
    .withColumn("review_date", F.to_date(F.to_timestamp("unix_timestamp_created")))
    .filter(F.col("review_date").isNotNull())
    .groupBy("appid", "review_date")
    .agg(F.count("*").alias("daily_review_count"))
)

# rangeBetween uses epoch-seconds because the order column is cast to long.
hype_window = (
    Window
    .partitionBy("appid")
    .orderBy(F.col("review_date").cast("long"))
    .rangeBetween(-(6 * 86_400), 0)
)

hype_trend_df = (
    daily_counts_df
    .withColumn("avg_7d_reviews", F.round(F.avg("daily_review_count").over(hype_window), 2))
    .orderBy("appid", "review_date")
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
    .groupBy("appid")
    .agg(
        F.round(F.avg("weighted_vote_score"), 4).alias("game_avg_score"),
        F.count("*").alias("game_review_count"),
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
        on="appid",
        how="inner",
    )
    .select(
        "steamid", "appid", "weighted_vote_score",
        "game_avg_score", "playtime_forever",
    )
)

with timer("Q7 — Subquery (elite reviews in above-average games)"):
    print(f"  Matching reviews: {above_avg_elite_df.count():,}")
    print(f"  Global avg threshold: {global_avg:.4f}")


# =============================================================================
# QUERY 8 — BROADCAST JOIN
# Goal: Enrich the reviews with game names and genres from a small
#       app_metadata lookup table without triggering a shuffle.
#
# Why broadcast: the metadata table is tiny (< 1 MB). F.broadcast() tells
# Catalyst to copy it to every executor's memory as a hash table. Each
# executor performs a local lookup — zero bytes cross the network for
# the metadata side.
# =============================================================================

broadcast_joined_df = (
    reviews
    .join(F.broadcast(app_metadata_df), on="appid", how="inner")
    .select("appid", "app_name", "genre", "voted_up",
            "weighted_vote_score", "playtime_forever")
)

print("\n--- Q8 Execution Plan (Broadcast Join) ---")
broadcast_joined_df.explain(True)
# Look for: BroadcastHashJoin (not SortMergeJoin) and BroadcastExchange
# on the metadata side.

with timer("Q8 — Broadcast Join (reviews x app metadata)"):
    broadcast_joined_df.show(10, truncate=False)


# =============================================================================
# QUERY 9 — PARTITION PRUNING
# Goal: Analyse monthly review volumes and average scores for 2022–2023 only,
#       demonstrating how partitioned storage eliminates unnecessary I/O.
#
# How it works: when data is written with .write.partitionBy("review_year",
# "review_month"), filtering on review_year allows Spark to open only those
# directories — all other years are never read from disk.
# =============================================================================

pruned_df = (
    reviews
    .filter(F.col("review_year").isin(2022, 2023))
    .groupBy("review_year", "review_month")
    .agg(
        F.count("*").alias("monthly_reviews"),
        F.round(F.avg("weighted_vote_score"), 4).alias("avg_score"),
    )
    .orderBy("review_year", "review_month")
)

print("\n--- Q9 Execution Plan (Partition Pruning) ---")
pruned_df.explain(True)
# After writing with partitionBy and re-reading, the Physical Plan shows:
#   PartitionFilters: [review_year IN (2022, 2023)]

with timer("Q9 — Partition Pruning (2022–2023 monthly summary)"):
    pruned_df.show(24, truncate=False)


# =============================================================================
# QUERY 10 — SORT-MERGE JOIN
# Goal: Join the reviews with a user_profiles table on steamid to add
#       account-level context.
#
# Why Sort-Merge Join: neither side is small enough to broadcast.
# Spark sorts both tables by the join key, then walks them in lockstep.
# Pre-calling .repartition(200, "steamid") aligns the partitioning upfront
# so AQE can skip the re-partition Exchange on that side.
# =============================================================================

reviews_pre_partitioned = reviews.repartition(200, "steamid")

sort_merge_joined_df = (
    reviews_pre_partitioned
    .join(user_profiles_df, on="steamid", how="inner")
    .select(
        "steamid", "username", "account_age_days",
        "total_reviews", "appid",
        "voted_up", "weighted_vote_score", "playtime_forever",
    )
    .withColumn("is_veteran", F.col("account_age_days") >= 365)
)

print("\n--- Q10 Execution Plan (Sort-Merge Join) ---")
sort_merge_joined_df.explain(True)
# Look for: SortMergeJoin [steamid], [steamid], Inner
# Two Exchange hashpartitioning nodes (one per table).

with timer("Q10 — Sort-Merge Join (reviews x user profiles)"):
    sort_merge_joined_df.show(15, truncate=False)


# =============================================================================
# CLEANUP
# =============================================================================

elite_df.unpersist()

spark.stop()
