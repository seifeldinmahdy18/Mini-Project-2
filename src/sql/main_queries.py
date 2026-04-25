# main_queries.py
# Spark SQL API — 10 queries on 15.4M Steam Reviews
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

import time
from contextlib import contextmanager

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    LongType, IntegerType, StringType, BooleanType, FloatType,
)


# =============================================================================
# SPARK SESSION  (identical config to DataFrame version)
# =============================================================================

spark = (
    SparkSession.builder
    .appName("SteamReviews_SQL_Queries")
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
    StructField("review_id",               LongType(),    nullable=False),
    StructField("app_id",                  LongType(),    nullable=False),
    StructField("author_steamid",          LongType(),    nullable=True),
    StructField("language",                StringType(),  nullable=True),
    StructField("review",                  StringType(),  nullable=True),
    StructField("timestamp_created",       LongType(),    nullable=True),
    StructField("recommended",             BooleanType(), nullable=True),
    StructField("votes_helpful",           IntegerType(), nullable=True),
    StructField("votes_funny",             IntegerType(), nullable=True),
    StructField("weighted_vote_score",     FloatType(),   nullable=True),
    StructField("author_playtime_forever", IntegerType(), nullable=True),
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
# DATA LOADING & TEMP VIEW REGISTRATION
# review_year / review_month are derived once in the base DataFrame so every
# SQL query can reference them as plain columns without repeating the expression.
# =============================================================================

DATA_PATH = "data/steam_reviews.parquet"

reviews = (
    spark.read
    .schema(REVIEW_SCHEMA)
    .parquet(DATA_PATH)
    .withColumn("review_year",  F.year(F.to_timestamp("timestamp_created")))
    .withColumn("review_month", F.month(F.to_timestamp("timestamp_created")))
)
reviews.createOrReplaceTempView("reviews")

# Small lookup tables — registered as views so SQL can reference them directly.
app_metadata_df = spark.createDataFrame(
    [
        (440,     "Team Fortress 2",   "Action"),
        (570,     "Dota 2",            "Strategy"),
        (730,     "CS:GO",             "FPS"),
        (578080,  "PUBG",              "Battle Royale"),
        (1172620, "Sea of Thieves",    "Adventure"),
        (1091500, "Cyberpunk 2077",    "RPG"),
        (945360,  "Among Us",         "Social Deduction"),
        (1422450, "Vampire Survivors", "Rogue-like"),
    ],
    schema=APP_META_SCHEMA,
)
app_metadata_df.createOrReplaceTempView("app_metadata")

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
    review_id,
    app_id,
    author_steamid,
    language,
    review,
    timestamp_created,
    votes_helpful,
    weighted_vote_score,
    author_playtime_forever,
    review_year
FROM reviews
WHERE author_playtime_forever >= 3000
  AND weighted_vote_score     >= 0.7
  AND recommended             = TRUE
  AND LENGTH(review)          >= 100
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
# Goal: Compute average helpfulness, review count, and total helpful votes
#       per game (only games with >= 50 reviews).
#
# Catalyst behaviour: GROUP BY triggers a partial HashAggregate on each
# executor before the shuffle. Only pre-aggregated partial results cross the
# network, reducing shuffle bytes by 80–90%. HAVING is pushed into a Filter
# node above the final aggregate — identical to the DataFrame .filter() call.
# =============================================================================

Q2_SQL = """
SELECT
    app_id,
    ROUND(AVG(weighted_vote_score),    4) AS avg_helpfulness,
    COUNT(review_id)                      AS total_reviews,
    ROUND(AVG(author_playtime_forever), 2) AS avg_playtime_mins,
    SUM(votes_helpful)                    AS total_votes_helpful
FROM reviews
GROUP BY app_id
HAVING COUNT(review_id) >= 50
"""

per_game_stats_sql_df = spark.sql(Q2_SQL)
per_game_stats_sql_df.createOrReplaceTempView("per_game_stats")

with timer("Q2 — Aggregation (avg helpfulness per game)"):
    per_game_stats_sql_df.show(10, truncate=False)


# =============================================================================
# QUERY 3 — MULTI-ATTRIBUTE GROUPING
# Goal: Group by language AND recommended to analyse cross-community sentiment.
#
# Catalyst behaviour: multi-column GROUP BY produces a wider shuffle key.
# AQE detects post-shuffle partition sizes and coalesces small tasks,
# avoiding the overhead of many near-empty reduce tasks — same as DataFrame.
# STDDEV_POP is pushed into the partial aggregate phase where possible.
# =============================================================================

Q3_SQL = """
SELECT
    language,
    recommended,
    COUNT(review_id)                           AS review_count,
    ROUND(AVG(weighted_vote_score),    4)      AS avg_helpfulness,
    ROUND(AVG(author_playtime_forever), 2)     AS avg_playtime_mins,
    ROUND(STDDEV_POP(weighted_vote_score), 4)  AS score_std_dev
FROM reviews
WHERE language    IS NOT NULL
  AND recommended IS NOT NULL
GROUP BY language, recommended
ORDER BY review_count DESC
"""

by_language_sentiment_sql_df = spark.sql(Q3_SQL)

with timer("Q3 — Multi-Attribute Grouping (language x recommendation)"):
    by_language_sentiment_sql_df.show(20, truncate=False)


# =============================================================================
# QUERY 4 — SORTING & RANKING
# Goal: Top 10 games by composite engagement score:
#         engagement = total_reviews * avg_helpfulness * log(avg_playtime + 1)
#
# Catalyst behaviour: ORDER BY + LIMIT is rewritten as TakeOrderedAndProject.
# Each executor maintains a local top-K heap; only 10 rows per executor travel
# to the driver — same optimisation as the DataFrame .orderBy().limit().
# The per_game_stats view is already materialised (no re-scan).
# =============================================================================

Q4_SQL = """
SELECT
    app_id,
    avg_helpfulness,
    total_reviews,
    avg_playtime_mins,
    total_votes_helpful,
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
# Catalyst behaviour: RANK() OVER (PARTITION BY app_id ...) generates an
# Exchange hashpartitioning(app_id) shuffle so all reviews for the same game
# share one executor. AQE can split skewed partitions (games with millions of
# reviews) across multiple tasks. The outer WHERE rank = 1 prunes rows after
# the window is applied.
# =============================================================================

Q5_SQL = """
SELECT
    app_id,
    review_id,
    author_steamid,
    language,
    recommended,
    weighted_vote_score,
    votes_helpful,
    author_playtime_forever,
    influence_rank
FROM (
    SELECT
        app_id,
        review_id,
        author_steamid,
        language,
        recommended,
        weighted_vote_score,
        votes_helpful,
        author_playtime_forever,
        RANK() OVER (
            PARTITION BY app_id
            ORDER BY weighted_vote_score DESC, votes_helpful DESC
        ) AS influence_rank
    FROM reviews
    WHERE weighted_vote_score IS NOT NULL
) ranked
WHERE influence_rank = 1
"""

top_review_per_game_sql_df = spark.sql(Q5_SQL)

print("\n--- Q5 Execution Plan (Window Rank) ---")
top_review_per_game_sql_df.explain(True)
# Look for: Window [rank() OVER (PARTITION BY app_id ORDER BY ...)]
#           Exchange hashpartitioning(app_id, 200)

with timer("Q5 — Window Rank (top review per game)"):
    print(f"  Games with a ranked top review: {top_review_per_game_sql_df.count():,}")


# =============================================================================
# QUERY 6 — WINDOW FUNCTION: MOVING AVERAGE
# Goal: 7-day rolling average of daily review volume per game.
#
# Catalyst behaviour: the inner CTE pre-aggregates to daily counts first,
# then the ROWS BETWEEN 6 PRECEDING AND CURRENT ROW window is applied on
# the smaller result. This avoids running a range window over 15.4M raw rows
# (which would cause large memory spill). Catalyst rewrites ROWS-based windows
# more efficiently than RANGE-based ones when the order column is not a date.
#
# Note: ROWS BETWEEN 6 PRECEDING AND CURRENT ROW counts 7 rows (including the
# current row), which is equivalent to the rangeBetween(-6*86400, 0) in the
# DataFrame version. Both capture the current day plus the 6 preceding days.
# =============================================================================

Q6_SQL = """
WITH daily_counts AS (
    SELECT
        app_id,
        TO_DATE(TO_TIMESTAMP(timestamp_created)) AS review_date,
        COUNT(review_id)                          AS daily_review_count
    FROM reviews
    WHERE TO_DATE(TO_TIMESTAMP(timestamp_created)) IS NOT NULL
    GROUP BY app_id, TO_DATE(TO_TIMESTAMP(timestamp_created))
)
SELECT
    app_id,
    review_date,
    daily_review_count,
    ROUND(
        AVG(daily_review_count) OVER (
            PARTITION BY app_id
            ORDER BY review_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        2
    ) AS avg_7d_reviews
FROM daily_counts
ORDER BY app_id, review_date
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
# dataset is not read twice. The EXISTS / IN subquery on per-game averages is
# rewritten as a semi-join or hash join depending on cardinality statistics.
# Look for a "Subquery" branch in the Physical Plan.
# =============================================================================

Q7_SQL = """
SELECT
    e.review_id,
    e.app_id,
    e.author_steamid,
    e.language,
    e.weighted_vote_score,
    g.game_avg_score,
    e.author_playtime_forever
FROM elite_reviews e
INNER JOIN (
    SELECT
        app_id,
        ROUND(AVG(weighted_vote_score), 4) AS game_avg_score,
        COUNT(review_id)                   AS game_review_count
    FROM reviews
    WHERE weighted_vote_score IS NOT NULL
    GROUP BY app_id
    HAVING COUNT(review_id) >= 20
       AND ROUND(AVG(weighted_vote_score), 4) > (
               SELECT AVG(weighted_vote_score)
               FROM reviews
               WHERE weighted_vote_score IS NOT NULL
           )
) g ON e.app_id = g.app_id
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
# Catalyst behaviour: /*+ BROADCAST(app_metadata) */ tells the planner to
# use BroadcastHashJoin regardless of the autoBroadcastJoinThreshold setting.
# The metadata table is copied to every executor as a JVM-side hash table.
# Each executor performs a local lookup — zero bytes cross the network for the
# metadata side. Compare to a Sort-Merge Join where both sides shuffle.
# =============================================================================

Q8_SQL = """
SELECT /*+ BROADCAST(app_metadata) */
    r.app_id,
    a.app_name,
    a.genre,
    r.review_id,
    r.recommended,
    r.weighted_vote_score,
    r.author_playtime_forever
FROM reviews r
INNER JOIN app_metadata a ON r.app_id = a.app_id
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
    COUNT(review_id)                      AS monthly_reviews,
    ROUND(AVG(weighted_vote_score), 4)    AS avg_score
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
# Goal: Join reviews with user_profiles on author_steamid.
#
# Catalyst behaviour: /*+ MERGE(user_profiles) */ forces SortMergeJoin even
# if user_profiles would otherwise qualify for broadcast. Both sides are
# shuffled and sorted on author_steamid, then walked in lockstep. With
# pre-repartitioning, AQE can detect that one side is already hash-partitioned
# on the join key and skip the re-partition Exchange on that side.
# The Physical Plan shows:
#   SortMergeJoin [author_steamid], [author_steamid], Inner
# =============================================================================

Q10_SQL = """
SELECT /*+ MERGE(user_profiles) */
    r.author_steamid,
    u.username,
    u.account_age_days,
    u.total_reviews,
    r.app_id,
    r.review_id,
    r.recommended,
    r.weighted_vote_score,
    r.author_playtime_forever,
    (u.account_age_days >= 365) AS is_veteran_account
FROM reviews r
INNER JOIN user_profiles u ON r.author_steamid = u.author_steamid
"""

sort_merge_joined_sql_df = spark.sql(Q10_SQL)

print("\n--- Q10 Execution Plan (Sort-Merge Join) ---")
sort_merge_joined_sql_df.explain(True)
# Look for: SortMergeJoin [author_steamid], [author_steamid], Inner
#           Two Exchange hashpartitioning nodes (one per table).

with timer("Q10 — Sort-Merge Join (reviews x user profiles)"):
    sort_merge_joined_sql_df.show(15, truncate=False)


# =============================================================================
# CLEANUP
# =============================================================================

elite_sql_df.unpersist()

spark.stop()
