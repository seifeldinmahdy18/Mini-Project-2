# main_queries.py
# Spark SQL API — 10 queries on Steam Reviews
# Dataset: kaggle.com/datasets/forgemaster/steam-reviews-dataset
#
# Spark SQL vs DataFrame trade-offs (global):
#   - SQL strings are parsed → analysed → Catalyst-optimised → codegen'd,
#     identical pipeline to the DataFrame API. The physical plans are the same.
#   - SQL hints (/*+ BROADCAST(...) */, /*+ MERGE(...) */) give the same
#     control as F.broadcast() or DataFrame join hints.
#   - Subqueries inside SQL strings are rewritten by Catalyst as scalar
#     subqueries / semi-joins — no extra Python round-trips.
#   - Temp views are metadata-only; the underlying data is not copied.

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
from pyspark.sql.types import (
    StructType, StructField,
    LongType, IntegerType, StringType, FloatType,
)


# =============================================================================
# SPARK SESSION  (identical config to DataFrame version)
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
# =============================================================================

REVIEW_SCHEMA = StructType([
    StructField("steamid",                LongType(),    nullable=False),
    StructField("appid",                  IntegerType(), nullable=False),
    StructField("voted_up",               StringType(),  nullable=True),
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
# DATA LOADING & TEMP VIEW REGISTRATION
# voted_up is cast to boolean immediately after loading.
# review_year / review_month are derived once so every SQL query can reference
# them as plain columns without repeating the expression.
# =============================================================================

DATA_PATH = "data/steam_reviews.parquet"

reviews = (
    spark.read
    .schema(REVIEW_SCHEMA)
    .parquet(DATA_PATH)
    .withColumn("voted_up",     F.lower(F.col("voted_up")).cast("boolean"))
    .withColumn("review_year",  F.year(F.to_timestamp("unix_timestamp_created")))
    .withColumn("review_month", F.month(F.to_timestamp("unix_timestamp_created")))
)
reviews.createOrReplaceTempView("reviews")

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
app_metadata_df.createOrReplaceTempView("app_metadata")

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
user_profiles_df.createOrReplaceTempView("user_profiles")


# =============================================================================
# TIMER UTILITY
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
# Catalyst behaviour: all four WHERE predicates are fused into a single
# compound Filter node in the Optimized Logical Plan, then pushed into the
# Parquet scan as PushedFilters. Numeric predicates skip entire row-groups
# before data enters JVM memory — identical to the DataFrame version.
# =============================================================================

Q1_SQL = """
SELECT
    steamid,
    appid,
    voted_up,
    votes_up,
    weighted_vote_score,
    playtime_forever,
    review,
    unix_timestamp_created,
    review_year
FROM reviews
WHERE playtime_forever    >= 3000
  AND weighted_vote_score >= 0.7
  AND voted_up            = TRUE
  AND LENGTH(review)      >= 100
"""

elite_sql_df = spark.sql(Q1_SQL)

print("\n--- Q1 Execution Plan (Complex Filter) ---")
elite_sql_df.explain(True)
# In Physical Plan look for:
#   PushedFilters: [IsNotNull(...), GreaterThanOrEqual(...), ...]

# Cache for reuse in Q7.
elite_sql_df.createOrReplaceTempView("elite_reviews")
elite_sql_df.cache()

with timer("Q1 — Complex Filter"):
    print(f"  Elite reviewers: {elite_sql_df.count():,}")


# =============================================================================
# QUERY 2 — AGGREGATION
# Goal: Compute average helpfulness, review count, avg playtime, and total
#       votes_up per game (only games with >= 50 reviews).
#
# Catalyst behaviour: GROUP BY triggers a partial HashAggregate on each
# executor before the shuffle. Only pre-aggregated partial results cross the
# network, reducing shuffle bytes by 80–90%. HAVING is pushed into a Filter
# node above the final aggregate.
# =============================================================================

Q2_SQL = """
SELECT
    appid,
    ROUND(AVG(weighted_vote_score), 4) AS avg_helpfulness,
    COUNT(*)                           AS total_reviews,
    ROUND(AVG(playtime_forever),    2) AS avg_playtime_mins,
    SUM(votes_up)                      AS total_votes_up
FROM reviews
GROUP BY appid
HAVING COUNT(*) >= 50
"""

per_game_stats_sql_df = spark.sql(Q2_SQL)
per_game_stats_sql_df.createOrReplaceTempView("per_game_stats")

with timer("Q2 — Aggregation (avg helpfulness per game)"):
    per_game_stats_sql_df.show(10, truncate=False)


# =============================================================================
# QUERY 3 — MULTI-ATTRIBUTE GROUPING
# Goal: Group by appid AND voted_up to analyse per-game sentiment.
#
# Catalyst behaviour: multi-column GROUP BY produces a wider shuffle key.
# AQE detects post-shuffle partition sizes and coalesces small tasks,
# avoiding the overhead of many near-empty reduce tasks.
# STDDEV_POP is pushed into the partial aggregate phase where possible.
# =============================================================================

Q3_SQL = """
SELECT
    appid,
    voted_up,
    COUNT(*)                              AS review_count,
    ROUND(AVG(weighted_vote_score),    4) AS avg_helpfulness,
    ROUND(AVG(playtime_forever),       2) AS avg_playtime_mins,
    ROUND(STDDEV_POP(weighted_vote_score), 4) AS score_std_dev
FROM reviews
WHERE voted_up IS NOT NULL
  AND appid    IS NOT NULL
GROUP BY appid, voted_up
ORDER BY review_count DESC
"""

by_app_sentiment_sql_df = spark.sql(Q3_SQL)

with timer("Q3 — Multi-Attribute Grouping (appid x voted_up)"):
    by_app_sentiment_sql_df.show(20, truncate=False)


# =============================================================================
# QUERY 4 — SORTING & RANKING
# Goal: Top 10 games by composite engagement score:
#         engagement = total_reviews * avg_helpfulness * log(avg_playtime + 1)
#
# Catalyst behaviour: ORDER BY + LIMIT is rewritten as TakeOrderedAndProject.
# Each executor maintains a local top-K heap; only 10 rows per executor travel
# to the driver. The per_game_stats view is already materialised (no re-scan).
# =============================================================================

Q4_SQL = """
SELECT
    appid,
    avg_helpfulness,
    total_reviews,
    avg_playtime_mins,
    total_votes_up,
    ROUND(total_reviews * avg_helpfulness * LN(avg_playtime_mins + 1), 2) AS engagement_score
FROM per_game_stats
ORDER BY engagement_score DESC
LIMIT 10
"""

top10_games_sql_df = spark.sql(Q4_SQL)

with timer("Q4 — Sorting & Ranking (top 10 games by engagement)"):
    top10_games_sql_df.show(10, truncate=False)


# =============================================================================
# QUERY 5 — WINDOW FUNCTION: RANKING
# Goal: Keep the single most influential review per game.
#
# Catalyst behaviour: RANK() OVER (PARTITION BY appid ...) generates an
# Exchange hashpartitioning(appid) shuffle so all reviews for the same game
# share one executor. AQE can split skewed partitions across multiple tasks.
# The outer WHERE filters to rank = 1 after the window is applied.
# =============================================================================

Q5_SQL = """
SELECT
    appid,
    steamid,
    voted_up,
    weighted_vote_score,
    votes_up,
    playtime_forever,
    influence_rank
FROM (
    SELECT
        appid,
        steamid,
        voted_up,
        weighted_vote_score,
        votes_up,
        playtime_forever,
        RANK() OVER (
            PARTITION BY appid
            ORDER BY weighted_vote_score DESC, votes_up DESC
        ) AS influence_rank
    FROM reviews
    WHERE weighted_vote_score IS NOT NULL
) ranked
WHERE influence_rank = 1
"""

top_review_per_game_sql_df = spark.sql(Q5_SQL)

print("\n--- Q5 Execution Plan (Window Rank) ---")
top_review_per_game_sql_df.explain(True)
# Look for: Window [rank() OVER (PARTITION BY appid ORDER BY ...)]
#           Exchange hashpartitioning(appid, 200)

with timer("Q5 — Window Rank (top review per game)"):
    print(f"  Games with a ranked top review: {top_review_per_game_sql_df.count():,}")


# =============================================================================
# QUERY 6 — WINDOW FUNCTION: MOVING AVERAGE
# Goal: 7-day rolling average of daily review volume per game.
#
# Catalyst behaviour: the inner CTE pre-aggregates to daily counts first,
# then ROWS BETWEEN 6 PRECEDING AND CURRENT ROW is applied on the smaller
# result. This avoids running a range window over the full raw dataset.
# ROWS-based windows are more efficient than RANGE-based when ordering by date.
# =============================================================================

Q6_SQL = """
WITH daily_counts AS (
    SELECT
        appid,
        TO_DATE(TO_TIMESTAMP(unix_timestamp_created)) AS review_date,
        COUNT(*)                                       AS daily_review_count
    FROM reviews
    WHERE TO_DATE(TO_TIMESTAMP(unix_timestamp_created)) IS NOT NULL
    GROUP BY appid, TO_DATE(TO_TIMESTAMP(unix_timestamp_created))
)
SELECT
    appid,
    review_date,
    daily_review_count,
    ROUND(
        AVG(daily_review_count) OVER (
            PARTITION BY appid
            ORDER BY review_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        2
    ) AS avg_7d_reviews
FROM daily_counts
ORDER BY appid, review_date
"""

hype_trend_sql_df = spark.sql(Q6_SQL)

with timer("Q6 — Moving Average (7-day review trend)"):
    hype_trend_sql_df.show(15, truncate=False)


# =============================================================================
# QUERY 7 — NESTED / SUBQUERY LOGIC
# Goal: Return elite reviews that belong to games whose average helpfulness
#       exceeds the global average.
#
# Catalyst behaviour: the inner scalar subquery (global AVG) is computed in a
# separate scan pass and its result is broadcast to the outer filter — the
# dataset is not read twice. The per-game HAVING subquery is rewritten as a
# semi-join or hash join depending on cardinality statistics. Look for a
# "Subquery" branch in the Physical Plan.
# =============================================================================

Q7_SQL = """
SELECT
    e.steamid,
    e.appid,
    e.weighted_vote_score,
    g.game_avg_score,
    e.playtime_forever
FROM elite_reviews e
INNER JOIN (
    SELECT
        appid,
        ROUND(AVG(weighted_vote_score), 4) AS game_avg_score,
        COUNT(*)                           AS game_review_count
    FROM reviews
    WHERE weighted_vote_score IS NOT NULL
    GROUP BY appid
    HAVING COUNT(*) >= 20
       AND ROUND(AVG(weighted_vote_score), 4) > (
               SELECT AVG(weighted_vote_score)
               FROM reviews
               WHERE weighted_vote_score IS NOT NULL
           )
) g ON e.appid = g.appid
"""

above_avg_elite_sql_df = spark.sql(Q7_SQL)

print("\n--- Q7 Execution Plan (Subquery) ---")
above_avg_elite_sql_df.explain(True)
# Look for a "Subquery" branch — scalar subquery for the global average.

with timer("Q7 — Subquery (elite reviews in above-average games)"):
    total = above_avg_elite_sql_df.count()
    global_avg = spark.sql(
        "SELECT AVG(weighted_vote_score) FROM reviews WHERE weighted_vote_score IS NOT NULL"
    ).collect()[0][0]
    print(f"  Matching reviews: {total:,}")
    print(f"  Global avg threshold: {global_avg:.4f}")


# =============================================================================
# QUERY 8 — BROADCAST JOIN
# Goal: Enrich reviews with game name and genre from a tiny lookup table.
#
# Catalyst behaviour: /*+ BROADCAST(app_metadata) */ tells the planner to use
# BroadcastHashJoin regardless of the autoBroadcastJoinThreshold setting.
# The metadata table is copied to every executor as a JVM-side hash table.
# Each executor performs a local lookup — zero bytes cross the network for
# the metadata side.
# =============================================================================

Q8_SQL = """
SELECT /*+ BROADCAST(app_metadata) */
    r.appid,
    a.app_name,
    a.genre,
    r.voted_up,
    r.weighted_vote_score,
    r.playtime_forever
FROM reviews r
INNER JOIN app_metadata a ON r.appid = a.appid
"""

broadcast_joined_sql_df = spark.sql(Q8_SQL)

print("\n--- Q8 Execution Plan (Broadcast Join) ---")
broadcast_joined_sql_df.explain(True)
# Look for: BroadcastHashJoin (not SortMergeJoin) and BroadcastExchange
# on the app_metadata side.

with timer("Q8 — Broadcast Join (reviews x app metadata)"):
    broadcast_joined_sql_df.show(10, truncate=False)


# =============================================================================
# QUERY 9 — PARTITION PRUNING
# Goal: Monthly review volumes and average scores for 2022–2023.
#
# Catalyst behaviour: WHERE review_year IN (2022, 2023) is pushed into the
# Parquet scan as a PartitionFilter. When the data is stored with
# partitionBy("review_year", "review_month"), Spark only opens those
# subdirectories — all other years are never read from disk.
# The Physical Plan shows:
#   PartitionFilters: [review_year IN (2022, 2023)]
# =============================================================================

Q9_SQL = """
SELECT
    review_year,
    review_month,
    COUNT(*)                           AS monthly_reviews,
    ROUND(AVG(weighted_vote_score), 4) AS avg_score
FROM reviews
WHERE review_year IN (2022, 2023)
GROUP BY review_year, review_month
ORDER BY review_year, review_month
"""

pruned_sql_df = spark.sql(Q9_SQL)

print("\n--- Q9 Execution Plan (Partition Pruning) ---")
pruned_sql_df.explain(True)
# After writing with partitionBy and re-reading:
#   PartitionFilters: [review_year IN (2022, 2023)]

with timer("Q9 — Partition Pruning (2022–2023 monthly summary)"):
    pruned_sql_df.show(24, truncate=False)


# =============================================================================
# QUERY 10 — SORT-MERGE JOIN
# Goal: Join reviews with user_profiles on steamid.
#
# Catalyst behaviour: /*+ MERGE(user_profiles) */ forces SortMergeJoin even
# if user_profiles would otherwise qualify for broadcast. Both sides are
# shuffled and sorted on steamid, then walked in lockstep. With
# pre-repartitioning, AQE can detect matching partitioning and skip the
# re-partition Exchange on that side.
# The Physical Plan shows:
#   SortMergeJoin [steamid], [steamid], Inner
# =============================================================================

Q10_SQL = """
SELECT /*+ MERGE(user_profiles) */
    r.steamid,
    u.username,
    u.account_age_days,
    u.total_reviews,
    r.appid,
    r.voted_up,
    r.weighted_vote_score,
    r.playtime_forever,
    (u.account_age_days >= 365) AS is_veteran
FROM reviews r
INNER JOIN user_profiles u ON r.steamid = u.steamid
"""

sort_merge_joined_sql_df = spark.sql(Q10_SQL)

print("\n--- Q10 Execution Plan (Sort-Merge Join) ---")
sort_merge_joined_sql_df.explain(True)
# Look for: SortMergeJoin [steamid], [steamid], Inner
#           Two Exchange hashpartitioning nodes (one per table).

with timer("Q10 — Sort-Merge Join (reviews x user profiles)"):
    sort_merge_joined_sql_df.show(15, truncate=False)


# =============================================================================
# CLEANUP
# =============================================================================

elite_sql_df.unpersist()

spark.stop()
