# main_queries.py
# PySpark RDD API — 10 queries on Steam Reviews
# Dataset: kaggle.com/datasets/forgemaster/steam-reviews-dataset
#
# RDD vs DataFrame trade-offs (global):
#   - No Catalyst optimizer: every transformation is opaque Python/JVM bytecode.
#   - No predicate pushdown: Parquet row-groups are fully deserialized before filtering.
#   - No partial aggregation rewrite: reduceByKey does combine locally, but no cost-based
#     join selection, no AQE skew splitting, no broadcast auto-detection.
#   - Full Row deserialization to Python objects on every record touch.

import os
os.environ["JAVA_HOME"] = "/usr/local/sdkman/candidates/java/21.0.10-ms"
os.environ["JAVA_TOOL_OPTIONS"] = (
    "-Djava.security.manager=allow "
    "--add-opens=java.base/javax.security.auth=ALL-UNNAMED"
)
import math
import time
import datetime as _dt
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
sc = spark.sparkContext


# =============================================================================
# SCHEMAS  (used only to read Parquet via the DataFrame reader, then .rdd)
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

USER_PROFILE_SCHEMA = StructType([
    StructField("steamid",          LongType(),    nullable=False),
    StructField("username",         StringType(),  nullable=True),
    StructField("account_age_days", IntegerType(), nullable=True),
    StructField("total_reviews",    IntegerType(), nullable=True),
])


# =============================================================================
# DATA LOADING
# spark.read.parquet gives us the schema-aware DataFrame; .rdd converts it to
# a Row-based RDD. voted_up is cast to boolean and time fields are derived here
# via a plain Python map — no Catalyst column derivation.
# =============================================================================

DATA_PATH = "data/steam_reviews.parquet"


def _parse_voted_up(val):
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"


def _add_time_fields(row):
    ts = row["unix_timestamp_created"]
    try:
        dt    = _dt.datetime.utcfromtimestamp(ts) if ts is not None else None
        year  = dt.year  if dt else None
        month = dt.month if dt else None
    except (OSError, OverflowError, ValueError):
        year = month = None
    return (
        row["steamid"],                          # 0
        row["appid"],                            # 1
        _parse_voted_up(row["voted_up"]),        # 2
        row["votes_up"],                         # 3
        row["votes_funny"],                      # 4
        row["weighted_vote_score"],              # 5
        row["playtime_forever"],                 # 6
        row["playtime_at_review"],               # 7
        row["num_games_owned"],                  # 8
        row["num_reviews"],                      # 9
        row["review"],                           # 10
        row["unix_timestamp_created"],           # 11
        row["unix_timestamp_updated"],           # 12
        year,                                    # 13  review_year
        month,                                   # 14  review_month
    )


# Field index constants
F_STEAMID          = 0
F_APPID            = 1
F_VOTED_UP         = 2
F_VOTES_UP         = 3
F_VOTES_FUNNY      = 4
F_WVS              = 5   # weighted_vote_score
F_PLAYTIME         = 6   # playtime_forever
F_PLAYTIME_AT_REV  = 7
F_NUM_GAMES        = 8
F_NUM_REVIEWS      = 9
F_REVIEW           = 10
F_TS_CREATED       = 11
F_TS_UPDATED       = 12
F_REVIEW_YEAR      = 13
F_REVIEW_MONTH     = 14

reviews_rdd = (
    spark.read
    .schema(REVIEW_SCHEMA)
    .parquet(DATA_PATH)
    .rdd
    .map(_add_time_fields)
)


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
# RDD cost: no predicate pushdown. All Parquet row-groups are fully decoded
# into JVM Row objects before any Python lambda filters them. The DataFrame
# version pushes numeric predicates into the Parquet reader and skips entire
# file chunks; the RDD version cannot.
# =============================================================================

elite_rdd = (
    reviews_rdd
    .filter(lambda r: r[F_PLAYTIME] is not None and r[F_PLAYTIME] >= 3_000)
    .filter(lambda r: r[F_WVS] is not None and r[F_WVS] >= 0.7)
    .filter(lambda r: r[F_VOTED_UP] is True)
    .filter(lambda r: r[F_REVIEW] is not None and len(r[F_REVIEW]) >= 100)
    .map(lambda r: (
        r[F_STEAMID], r[F_APPID], r[F_VOTED_UP], r[F_VOTES_UP],
        r[F_WVS], r[F_PLAYTIME], r[F_REVIEW], r[F_TS_CREATED], r[F_REVIEW_YEAR],
    ))
)

# Cache because Q7 reuses this RDD.
elite_rdd.cache()

with timer("Q1 — Complex Filter"):
    print(f"  Elite reviewers: {elite_rdd.count():,}")


# =============================================================================
# QUERY 2 — AGGREGATION
# Goal: Compute average helpfulness, review count, avg playtime, and total
#       votes_up per game (only games with >= 50 reviews).
#
# RDD cost: reduceByKey performs a local combine (like a partial aggregate)
# before shuffling, so network bytes are reduced — but the plan is fixed at
# the API level. Catalyst would choose between hash-agg and sort-agg based on
# runtime statistics; here we always pay a full shuffle.
# =============================================================================

# Emit (appid, (wvs, playtime, votes_up, 1)) per review.
per_game_stats_rdd = (
    reviews_rdd
    .map(lambda r: (
        r[F_APPID],
        (
            r[F_WVS]      or 0.0,
            r[F_PLAYTIME] or 0,
            r[F_VOTES_UP] or 0,
            1,
        )
    ))
    .reduceByKey(lambda a, b: (
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
    ))
    .filter(lambda kv: kv[1][3] >= 50)
    .map(lambda kv: (
        kv[0],                                # appid
        round(kv[1][0] / kv[1][3], 4),        # avg_helpfulness
        kv[1][3],                             # total_reviews
        round(kv[1][1] / kv[1][3], 2),        # avg_playtime_mins
        kv[1][2],                             # total_votes_up
    ))
)

with timer("Q2 — Aggregation (avg helpfulness per game)"):
    for row in per_game_stats_rdd.take(10):
        print(f"  appid={row[0]}  avg_helpfulness={row[1]}  "
              f"total_reviews={row[2]}  avg_playtime={row[3]}  "
              f"total_votes_up={row[4]}")


# =============================================================================
# QUERY 3 — MULTI-ATTRIBUTE GROUPING
# Goal: Group by appid AND voted_up to analyse per-game sentiment.
#
# RDD cost: composite keys are Python tuples — Kryo cannot apply column-level
# encoders. The shuffle key is larger and the post-shuffle coalescing that
# AQE performs automatically must be accepted as-is.
# =============================================================================

by_app_sentiment_rdd = (
    reviews_rdd
    .filter(lambda r: r[F_VOTED_UP] is not None and r[F_APPID] is not None)
    .map(lambda r: (
        (r[F_APPID], r[F_VOTED_UP]),
        (
            r[F_WVS]      or 0.0,
            (r[F_WVS]     or 0.0) ** 2,   # for population std-dev
            r[F_PLAYTIME] or 0,
            1,
        )
    ))
    .reduceByKey(lambda a, b: (
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
    ))
    .map(lambda kv: (
        kv[0][0],                                               # appid
        kv[0][1],                                               # voted_up
        kv[1][3],                                               # review_count
        round(kv[1][0] / kv[1][3], 4),                         # avg_helpfulness
        round(kv[1][2] / kv[1][3], 2),                         # avg_playtime_mins
        round(
            math.sqrt(
                max(0.0, kv[1][1] / kv[1][3] - (kv[1][0] / kv[1][3]) ** 2)
            ), 4
        ),                                                      # score_std_dev
    ))
    .sortBy(lambda r: -r[2])
)

with timer("Q3 — Multi-Attribute Grouping (appid x voted_up)"):
    for row in by_app_sentiment_rdd.take(20):
        print(f"  appid={row[0]}  voted_up={row[1]}  count={row[2]}  "
              f"avg_help={row[3]}  avg_play={row[4]}  std={row[5]}")


# =============================================================================
# QUERY 4 — SORTING & RANKING
# Goal: Top 10 games by composite engagement score:
#         engagement = total_reviews * avg_helpfulness * log(avg_playtime + 1)
#
# RDD cost: no TakeOrderedAndProject optimisation. sortBy forces a full
# shuffle-sort across all partitions before take() reads the first result.
# The DataFrame version rewrites ORDER BY + LIMIT as a local top-K heap per
# executor, exchanging only K rows with the driver.
# =============================================================================

top10_games_rdd = (
    per_game_stats_rdd
    .map(lambda r: (
        r[0],                                          # appid
        r[1],                                          # avg_helpfulness
        r[2],                                          # total_reviews
        r[3],                                          # avg_playtime_mins
        r[4],                                          # total_votes_up
        round(r[2] * r[1] * math.log1p(r[3]), 2),     # engagement_score
    ))
    .sortBy(lambda r: -r[5])
)

with timer("Q4 — Sorting & Ranking (top 10 games by engagement)"):
    for row in top10_games_rdd.take(10):
        print(f"  appid={row[0]}  avg_help={row[1]}  total_reviews={row[2]}  "
              f"avg_play={row[3]}  engagement={row[5]}")


# =============================================================================
# QUERY 5 — WINDOW FUNCTION: RANKING  (manual implementation)
# Goal: Keep only the single most influential review per game.
#
# RDD cost: no native window operator. groupByKey collects all reviews for a
# game into a single Python list on one executor — large in-memory buffers for
# popular games, and cannot exploit AQE skew-split. The DataFrame Window API
# keeps data in JVM-side sorted buffers and can spill to disk gracefully.
# =============================================================================

top_review_per_game_rdd = (
    reviews_rdd
    .filter(lambda r: r[F_WVS] is not None)
    .map(lambda r: (
        r[F_APPID],
        (r[F_STEAMID], r[F_VOTED_UP], r[F_WVS], r[F_VOTES_UP], r[F_PLAYTIME]),
    ))
    .groupByKey()
    .flatMap(lambda kv: [
        (
            kv[0],                  # appid
            best[0],                # steamid
            best[1],                # voted_up
            best[2],                # weighted_vote_score
            best[3],                # votes_up
            best[4],                # playtime_forever
            1,                      # influence_rank
        )
        for best in [
            sorted(kv[1], key=lambda x: (-(x[2] or 0.0), -(x[3] or 0)))[0]
        ]
    ])
)

with timer("Q5 — Window Rank (top review per game)"):
    print(f"  Games with a ranked top review: {top_review_per_game_rdd.count():,}")


# =============================================================================
# QUERY 6 — WINDOW FUNCTION: MOVING AVERAGE  (manual sliding window)
# Goal: Compute a 7-day rolling average of daily review volume per game.
#
# RDD cost: same groupByKey OOM risk as Q5. The rolling window is computed in
# Python with a sorted list and a linear scan — O(D) per game where D is the
# number of distinct days. The DataFrame rangeBetween window is backed by a
# JVM-side sorted deque with O(1) slide and can spill to disk.
# =============================================================================

def _to_day_epoch(ts):
    if ts is None:
        return None
    try:
        return int(ts) // 86_400
    except (TypeError, ValueError):
        return None


def _rolling_avg(sorted_day_counts, window_days=7):
    days = sorted_day_counts
    n = len(days)
    left = 0
    running_sum = 0
    for right in range(n):
        day_r, cnt_r = days[right]
        running_sum += cnt_r
        while day_r - days[left][0] >= window_days:
            running_sum -= days[left][1]
            left += 1
        yield (day_r, round(running_sum / (right - left + 1), 2))


daily_counts_rdd = (
    reviews_rdd
    .map(lambda r: (_to_day_epoch(r[F_TS_CREATED]), r[F_APPID]))
    .filter(lambda r: r[0] is not None)
    .map(lambda r: ((r[1], r[0]), 1))
    .reduceByKey(lambda a, b: a + b)
    .map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))   # (appid, (day_epoch, count))
)

hype_trend_rdd = (
    daily_counts_rdd
    .groupByKey()
    .flatMap(lambda kv: (
        (kv[0], day_epoch, avg_7d)
        for day_epoch, avg_7d in _rolling_avg(
            sorted(kv[1], key=lambda x: x[0])
        )
    ))
    .sortBy(lambda r: (r[0], r[1]))
)

with timer("Q6 — Moving Average (7-day review trend)"):
    for row in hype_trend_rdd.take(15):
        print(f"  appid={row[0]}  day_epoch={row[1]}  avg_7d={row[2]}")


# =============================================================================
# QUERY 7 — NESTED / SUBQUERY LOGIC
# Goal: Return elite reviews in games whose average helpfulness exceeds the
#       global average.
#
# RDD cost: the global average requires a separate .mean() action — a full
# extra pass over the score RDD. This is exactly the second scan that Catalyst
# avoids by computing the scalar subquery in a single stage.
# =============================================================================

# Phase 1: global average (separate scan — unavoidable in RDD API)
global_avg: float = (
    reviews_rdd
    .map(lambda r: r[F_WVS])
    .filter(lambda v: v is not None)
    .mean()
)
print(f"\n  Global average helpfulness score: {global_avg:.4f}")

# Phase 2: per-game average (min 20 reviews)
per_game_avg_rdd = (
    reviews_rdd
    .filter(lambda r: r[F_WVS] is not None)
    .map(lambda r: (r[F_APPID], (r[F_WVS], 1)))
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    .filter(lambda kv: kv[1][1] >= 20)
    .map(lambda kv: (kv[0], round(kv[1][0] / kv[1][1], 4)))
    .filter(lambda kv: kv[1] > global_avg)
)

# Phase 3: join elite reviews with above-average games.
# elite fields: steamid[0], appid[1], voted_up[2], votes_up[3],
#               wvs[4], playtime[5], review[6], ts[7], year[8]
elite_keyed = elite_rdd.map(lambda r: (r[1], r))   # key=appid
above_avg_elite_rdd = (
    elite_keyed
    .join(per_game_avg_rdd)
    .map(lambda kv: (
        kv[1][0][0],   # steamid
        kv[0],         # appid
        kv[1][0][4],   # weighted_vote_score
        kv[1][1],      # game_avg_score
        kv[1][0][5],   # playtime_forever
    ))
)

with timer("Q7 — Subquery (elite reviews in above-average games)"):
    print(f"  Matching reviews: {above_avg_elite_rdd.count():,}")
    print(f"  Global avg threshold: {global_avg:.4f}")


# =============================================================================
# QUERY 8 — BROADCAST JOIN  (sc.broadcast)
# Goal: Enrich reviews with game name and genre using a tiny lookup table.
#
# RDD cost: sc.broadcast() ships the dict to each executor once — equivalent
# to BroadcastHashJoin. However, the broadcast variable is a plain Python dict
# (not a JVM-side hash table) so every lookup pays Python interpreter overhead
# instead of a JVM HashMap.get(). For 8 rows this is negligible, but at scale
# the JVM-side structure is orders of magnitude faster.
# =============================================================================

app_metadata_local = {
    10:      ("Counter-Strike",    "Action"),
    570:     ("Dota 2",            "Strategy"),
    730:     ("CS:GO",             "FPS"),
    578080:  ("PUBG",              "Battle Royale"),
    1172620: ("Sea of Thieves",    "Adventure"),
    1091500: ("Cyberpunk 2077",    "RPG"),
    945360:  ("Among Us",          "Social Deduction"),
    1422450: ("Vampire Survivors", "Rogue-like"),
}

app_meta_bc = sc.broadcast(app_metadata_local)

broadcast_joined_rdd = (
    reviews_rdd
    .filter(lambda r: r[F_APPID] in app_meta_bc.value)
    .map(lambda r: (
        r[F_APPID],
        app_meta_bc.value[r[F_APPID]][0],   # app_name
        app_meta_bc.value[r[F_APPID]][1],   # genre
        r[F_VOTED_UP],
        r[F_WVS],
        r[F_PLAYTIME],
    ))
)

with timer("Q8 — Broadcast Join (reviews x app metadata)"):
    for row in broadcast_joined_rdd.take(10):
        print(f"  appid={row[0]}  app_name={row[1]}  genre={row[2]}  "
              f"voted_up={row[3]}  wvs={row[4]}  playtime={row[5]}")


# =============================================================================
# QUERY 9 — PARTITION PRUNING  (filter before aggregation)
# Goal: Monthly review volumes and average scores for 2022–2023.
#
# RDD cost: there is no partition pruning in the RDD API. Even with an early
# filter on review_year, Spark must read and deserialise every Parquet
# row-group before the Python lambda can discard non-target years. The
# DataFrame version opens only the matching year directories when the data
# was written with partitionBy("review_year", "review_month").
# =============================================================================

pruned_rdd = (
    reviews_rdd
    .filter(lambda r: r[F_REVIEW_YEAR] in (2022, 2023))
    .map(lambda r: (
        (r[F_REVIEW_YEAR], r[F_REVIEW_MONTH]),
        (r[F_WVS] or 0.0, 1),
    ))
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    .map(lambda kv: (
        kv[0][0],                              # review_year
        kv[0][1],                              # review_month
        kv[1][1],                              # monthly_reviews
        round(kv[1][0] / kv[1][1], 4),         # avg_score
    ))
    .sortBy(lambda r: (r[0], r[1]))
)

with timer("Q9 — Partition Pruning (2022–2023 monthly summary)"):
    for row in pruned_rdd.take(24):
        print(f"  year={row[0]}  month={row[1]}  "
              f"monthly_reviews={row[2]}  avg_score={row[3]}")


# =============================================================================
# QUERY 10 — SORT-MERGE JOIN
# Goal: Join reviews with user_profiles on steamid.
#
# RDD cost: .join() on two RDDs always performs a full shuffle-sort (Exchange
# hashpartitioning on both sides) followed by a merge pass. The DataFrame
# pre-repartition optimisation (AQE detecting matching partitioning and
# skipping one Exchange) is not available here — both sides always shuffle.
# =============================================================================

user_profiles_local = [
    (76561198107294407, "NightStrike",  1200, 315),
    (76561198011733201, "VoidWalker",   890,  42),
    (76561198000000003, "CatalystKing", 2100, 178),
    (76561198000000004, "LagSlayer",    450,  9),
    (76561198000000005, "FrostByte",    3300, 501),
    (76561198000000006, "GankMaster",   760,  88),
    (76561198000000007, "SparkArcher",  1980, 234),
    (76561198000000008, "ManaVault",    550,  61),
]

user_profiles_rdd = sc.parallelize(user_profiles_local)

reviews_keyed       = reviews_rdd.map(lambda r: (r[F_STEAMID], r))
user_profiles_keyed = user_profiles_rdd.map(lambda u: (u[0], u))

sort_merge_joined_rdd = (
    reviews_keyed
    .join(user_profiles_keyed)
    .map(lambda kv: (
        kv[0],                              # steamid
        kv[1][1][1],                        # username
        kv[1][1][2],                        # account_age_days
        kv[1][1][3],                        # total_reviews (user)
        kv[1][0][F_APPID],                  # appid
        kv[1][0][F_VOTED_UP],               # voted_up
        kv[1][0][F_WVS],                    # weighted_vote_score
        kv[1][0][F_PLAYTIME],               # playtime_forever
        kv[1][1][2] >= 365,                 # is_veteran
    ))
)

with timer("Q10 — Sort-Merge Join (reviews x user profiles)"):
    for row in sort_merge_joined_rdd.take(15):
        print(f"  steamid={row[0]}  user={row[1]}  age={row[2]}  "
              f"appid={row[4]}  voted_up={row[5]}  wvs={row[6]}  veteran={row[8]}")


# =============================================================================
# CLEANUP
# =============================================================================

elite_rdd.unpersist()
app_meta_bc.destroy()

spark.stop()
