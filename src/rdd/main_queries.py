# main_queries.py
# PySpark RDD API — 10 queries on 15.4M Steam Reviews
# Dataset: kaggle.com/datasets/forgemaster/steam-reviews-dataset
#
# RDD vs DataFrame trade-offs (global):
#   - No Catalyst optimizer: every transformation is opaque Python/JVM bytecode.
#   - No predicate pushdown: Parquet row-groups are fully deserialized before filtering.
#   - No partial aggregation rewrite: reduceByKey does combine locally, but no cost-based
#     join selection, no AQE skew splitting, no broadcast auto-detection.
#   - Full Row deserialization to Python objects on every record touch.

import math
import time
from contextlib import contextmanager
from itertools import islice

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    LongType, IntegerType, StringType, BooleanType, FloatType,
)


# =============================================================================
# SPARK SESSION  (identical config to DataFrame version)
# =============================================================================

spark = (
    SparkSession.builder
    .appName("SteamReviews_RDD_Queries")
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
# DATA LOADING
# spark.read.parquet gives us the schema-aware DataFrame; .rdd converts it to
# a Row-based RDD so all subsequent operations use the RDD API exclusively.
# review_year / review_month are derived here once and stored on each Row via
# a plain Python map — no Catalyst column derivation.
# =============================================================================

DATA_PATH = "data/steam_reviews.parquet"

import datetime as _dt

def _add_time_fields(row):
    ts = row["timestamp_created"]
    try:
        dt = _dt.datetime.utcfromtimestamp(ts) if ts is not None else None
        year  = dt.year  if dt else None
        month = dt.month if dt else None
    except (OSError, OverflowError, ValueError):
        year = month = None
    return (
        row["review_id"],
        row["app_id"],
        row["author_steamid"],
        row["language"],
        row["review"],
        row["timestamp_created"],
        row["recommended"],
        row["votes_helpful"],
        row["votes_funny"],
        row["weighted_vote_score"],
        row["author_playtime_forever"],
        year,   # index 11 — review_year
        month,  # index 12 — review_month
    )

# Field index constants for readability
F_REVIEW_ID               = 0
F_APP_ID                  = 1
F_AUTHOR_STEAMID          = 2
F_LANGUAGE                = 3
F_REVIEW                  = 4
F_TIMESTAMP               = 5
F_RECOMMENDED             = 6
F_VOTES_HELPFUL           = 7
F_VOTES_FUNNY             = 8
F_WEIGHTED_VOTE_SCORE     = 9
F_PLAYTIME_FOREVER        = 10
F_REVIEW_YEAR             = 11
F_REVIEW_MONTH            = 12

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
# RDD cost: no predicate pushdown. All 15.4M Parquet row-groups are fully
# decoded into JVM Row objects before a single Python lambda filters them.
# The DataFrame version pushes numeric predicates into the Parquet reader
# and skips entire file chunks; the RDD version cannot.
# =============================================================================

# RDD fields selected: (review_id, app_id, author_steamid, language, review,
#                        timestamp_created, votes_helpful, weighted_vote_score,
#                        author_playtime_forever, review_year)
elite_rdd = (
    reviews_rdd
    .filter(lambda r: r[F_PLAYTIME_FOREVER] is not None and r[F_PLAYTIME_FOREVER] >= 3_000)
    .filter(lambda r: r[F_WEIGHTED_VOTE_SCORE] is not None and r[F_WEIGHTED_VOTE_SCORE] >= 0.7)
    .filter(lambda r: r[F_RECOMMENDED] is True)
    .filter(lambda r: r[F_REVIEW] is not None and len(r[F_REVIEW]) >= 100)
    .map(lambda r: (
        r[F_REVIEW_ID], r[F_APP_ID], r[F_AUTHOR_STEAMID], r[F_LANGUAGE],
        r[F_REVIEW], r[F_TIMESTAMP], r[F_VOTES_HELPFUL],
        r[F_WEIGHTED_VOTE_SCORE], r[F_PLAYTIME_FOREVER], r[F_REVIEW_YEAR],
    ))
)

# Cache because Q7 reuses this RDD.
elite_rdd.cache()

with timer("Q1 — Complex Filter"):
    print(f"  Elite reviewers: {elite_rdd.count():,}")


# =============================================================================
# QUERY 2 — AGGREGATION
# Goal: Compute average helpfulness, review count, and total helpful votes
#       per game (only games with >= 50 reviews).
#
# RDD cost: reduceByKey performs a local combine (like a partial aggregate)
# before shuffling, so network bytes are reduced — but the plan is fixed
# at the API level. Catalyst would choose between hash-agg and sort-agg
# based on runtime statistics; here we always pay a full shuffle.
# =============================================================================

# Emit (app_id, (score, playtime, votes_helpful, 1)) per review.
per_game_stats_rdd = (
    reviews_rdd
    .map(lambda r: (
        r[F_APP_ID],
        (
            r[F_WEIGHTED_VOTE_SCORE] or 0.0,
            r[F_PLAYTIME_FOREVER]    or 0,
            r[F_VOTES_HELPFUL]       or 0,
            1,
        )
    ))
    .reduceByKey(lambda a, b: (
        a[0] + b[0],   # sum of scores
        a[1] + b[1],   # sum of playtime
        a[2] + b[2],   # sum of votes_helpful
        a[3] + b[3],   # count
    ))
    .filter(lambda kv: kv[1][3] >= 50)
    .map(lambda kv: (
        kv[0],                                              # app_id
        round(kv[1][0] / kv[1][3], 4),                     # avg_helpfulness
        kv[1][3],                                           # total_reviews
        round(kv[1][1] / kv[1][3], 2),                     # avg_playtime_mins
        kv[1][2],                                           # total_votes_helpful
    ))
)

with timer("Q2 — Aggregation (avg helpfulness per game)"):
    for row in per_game_stats_rdd.take(10):
        print(f"  app_id={row[0]}  avg_helpfulness={row[1]}  "
              f"total_reviews={row[2]}  avg_playtime={row[3]}  "
              f"total_votes_helpful={row[4]}")


# =============================================================================
# QUERY 3 — MULTI-ATTRIBUTE GROUPING
# Goal: Group by language AND recommended to see how review sentiment and
#       helpfulness differ across language communities.
#
# RDD cost: composite keys are Python tuples — Kryo cannot apply column-level
# encoders. The shuffle key is larger and the post-shuffle coalescing that
# AQE performs automatically must be done manually or just accepted.
# =============================================================================

by_language_sentiment_rdd = (
    reviews_rdd
    .filter(lambda r: r[F_LANGUAGE] is not None and r[F_RECOMMENDED] is not None)
    .map(lambda r: (
        (r[F_LANGUAGE], r[F_RECOMMENDED]),
        (
            r[F_WEIGHTED_VOTE_SCORE] or 0.0,
            (r[F_WEIGHTED_VOTE_SCORE] or 0.0) ** 2,  # for std-dev via Welford-free formula
            r[F_PLAYTIME_FOREVER] or 0,
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
        kv[0][0],                                    # language
        kv[0][1],                                    # recommended
        kv[1][3],                                    # review_count
        round(kv[1][0] / kv[1][3], 4),               # avg_helpfulness
        round(kv[1][2] / kv[1][3], 2),               # avg_playtime_mins
        round(
            math.sqrt(
                max(0.0, kv[1][1] / kv[1][3] - (kv[1][0] / kv[1][3]) ** 2)
            ), 4
        ),                                           # score_std_dev (population)
    ))
    .sortBy(lambda r: -r[2])
)

with timer("Q3 — Multi-Attribute Grouping (language x recommendation)"):
    for row in by_language_sentiment_rdd.take(20):
        print(f"  lang={row[0]}  rec={row[1]}  count={row[2]}  "
              f"avg_help={row[3]}  avg_play={row[4]}  std={row[5]}")


# =============================================================================
# QUERY 4 — SORTING & RANKING
# Goal: Find the top 10 games by composite engagement score:
#         engagement = total_reviews * avg_helpfulness * log(avg_playtime + 1)
#
# RDD cost: no TakeOrderedAndProject optimisation. sortBy forces a full
# shuffle-sort across all partitions before top() can read the first result.
# The DataFrame version rewrites ORDER BY + LIMIT as a local top-K heap per
# executor, exchanging only K rows with the driver.
# =============================================================================

top10_games_rdd = (
    per_game_stats_rdd
    .map(lambda r: (
        r[0],                                                        # app_id
        r[1],                                                        # avg_helpfulness
        r[2],                                                        # total_reviews
        r[3],                                                        # avg_playtime_mins
        r[4],                                                        # total_votes_helpful
        round(r[2] * r[1] * math.log1p(r[3]), 2),                   # engagement_score
    ))
    .sortBy(lambda r: -r[5])
)

with timer("Q4 — Sorting & Ranking (top 10 games by engagement)"):
    for row in top10_games_rdd.take(10):
        print(f"  app_id={row[0]}  avg_help={row[1]}  total_reviews={row[2]}  "
              f"avg_play={row[3]}  engagement={row[5]}")


# =============================================================================
# QUERY 5 — WINDOW FUNCTION: RANKING  (manual implementation)
# Goal: Keep only the single most influential review per game.
#
# RDD cost: no native window operator. groupByKey collects all reviews for
# a game into a single Python list in memory on one executor — this creates
# large in-memory buffers for popular games and cannot exploit AQE skew-split.
# The DataFrame Window API keeps data in JVM-side sorted buffers and can
# spill to disk gracefully; groupByKey spills the whole Python list or OOMs.
# =============================================================================

top_review_per_game_rdd = (
    reviews_rdd
    .filter(lambda r: r[F_WEIGHTED_VOTE_SCORE] is not None)
    .map(lambda r: (
        r[F_APP_ID],
        (
            r[F_REVIEW_ID], r[F_AUTHOR_STEAMID], r[F_LANGUAGE],
            r[F_RECOMMENDED], r[F_WEIGHTED_VOTE_SCORE],
            r[F_VOTES_HELPFUL], r[F_PLAYTIME_FOREVER],
        )
    ))
    .groupByKey()
    .flatMap(lambda kv: (
        # Sort descending by (weighted_vote_score, votes_helpful); take rank-1 only.
        [
            (
                kv[0],                  # app_id
                best[0],                # review_id
                best[1],                # author_steamid
                best[2],                # language
                best[3],                # recommended
                best[4],                # weighted_vote_score
                best[5],                # votes_helpful
                best[6],                # author_playtime_forever
                1,                      # influence_rank
            )
            for best in [
                sorted(kv[1], key=lambda x: (-x[4], -(x[5] or 0)))[0]
            ]
        ]
    ))
)

with timer("Q5 — Window Rank (top review per game)"):
    print(f"  Games with a ranked top review: {top_review_per_game_rdd.count():,}")


# =============================================================================
# QUERY 6 — WINDOW FUNCTION: MOVING AVERAGE  (manual sliding window)
# Goal: Compute a 7-day rolling average of daily review volume per game.
#
# RDD cost: same groupByKey OOM risk as Q5. The rolling window is computed
# in Python with a sorted list and a linear scan — O(D) per game where D is
# the number of distinct days.  The DataFrame rangeBetween window is backed
# by a JVM-side sorted deque with O(1) slide and can spill to disk.
# =============================================================================

def _to_date_epoch(ts):
    """Return the Unix day number (floor of epoch / 86400) for a timestamp."""
    if ts is None:
        return None
    try:
        return int(ts) // 86_400
    except (TypeError, ValueError):
        return None


def _rolling_avg(sorted_day_counts, window_days=7):
    """
    Given [(day_epoch, count), ...] sorted ascending, yield
    (day_epoch, avg_7d) for each entry using a sliding sum.
    """
    days = sorted_day_counts  # already sorted
    n = len(days)
    left = 0
    running_sum = 0
    for right in range(n):
        day_r, cnt_r = days[right]
        running_sum += cnt_r
        # Evict days outside the 7-day window.
        while day_r - days[left][0] >= window_days:
            running_sum -= days[left][1]
            left += 1
        window_len = right - left + 1
        yield (day_r, round(running_sum / window_len, 2))


daily_counts_rdd = (
    reviews_rdd
    .map(lambda r: (_to_date_epoch(r[F_TIMESTAMP]), r[F_APP_ID], r[F_REVIEW_ID]))
    .filter(lambda r: r[0] is not None)
    .map(lambda r: ((r[1], r[0]), 1))           # key=(app_id, day_epoch)
    .reduceByKey(lambda a, b: a + b)
    .map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))  # key=app_id, val=(day, count)
)

hype_trend_rdd = (
    daily_counts_rdd
    .groupByKey()
    .flatMap(lambda kv: (
        (
            (kv[0], day_epoch, avg_7d)
            for day_epoch, avg_7d in _rolling_avg(
                sorted(kv[1], key=lambda x: x[0])
            )
        )
    ))
    .sortBy(lambda r: (r[0], r[1]))
)

with timer("Q6 — Moving Average (7-day review trend)"):
    for row in hype_trend_rdd.take(15):
        print(f"  app_id={row[0]}  day_epoch={row[1]}  avg_7d={row[2]}")


# =============================================================================
# QUERY 7 — NESTED / SUBQUERY LOGIC
# Goal: Return elite reviews in games whose average helpfulness exceeds the
#       global average.
#
# RDD cost: the global average requires a separate .mean() action that forces
# a full pass over the score RDD. This is a second scan of the dataset —
# exactly what Catalyst avoids by computing the scalar subquery in one stage.
# =============================================================================

# Phase 1: global average (separate scan — unavoidable in RDD API)
global_avg: float = (
    reviews_rdd
    .map(lambda r: r[F_WEIGHTED_VOTE_SCORE])
    .filter(lambda v: v is not None)
    .mean()
)
print(f"\n  Global average helpfulness score: {global_avg:.4f}")

# Phase 2: per-game average (min 20 reviews)
per_game_avg_rdd = (
    reviews_rdd
    .filter(lambda r: r[F_WEIGHTED_VOTE_SCORE] is not None)
    .map(lambda r: (r[F_APP_ID], (r[F_WEIGHTED_VOTE_SCORE], 1)))
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    .filter(lambda kv: kv[1][1] >= 20)
    .map(lambda kv: (kv[0], round(kv[1][0] / kv[1][1], 4)))   # (app_id, game_avg_score)
    .filter(lambda kv: kv[1] > global_avg)
)

# Phase 3: join elite reviews with above-average games.
# elite fields: (review_id[0], app_id[1], author_steamid[2], language[3],
#                review[4], timestamp[5], votes_helpful[6], wvs[7], playtime[8], year[9])
elite_keyed   = elite_rdd.map(lambda r: (r[1], r))    # key=app_id
above_avg_elite_rdd = (
    elite_keyed
    .join(per_game_avg_rdd)   # (app_id, (elite_row, game_avg_score))
    .map(lambda kv: (
        kv[1][0][0],   # review_id
        kv[0],         # app_id
        kv[1][0][2],   # author_steamid
        kv[1][0][3],   # language
        kv[1][0][7],   # weighted_vote_score
        kv[1][1],      # game_avg_score
        kv[1][0][8],   # author_playtime_forever
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
# (not a JVM-side hash table), so every lookup pays Python interpreter overhead
# instead of a JVM HashMap.get(). For 8 rows this is negligible, but at scale
# the JVM-side structure is orders of magnitude faster.
# =============================================================================

app_metadata_local = {
    440:     ("Team Fortress 2",   "Action"),
    570:     ("Dota 2",            "Strategy"),
    730:     ("CS:GO",             "FPS"),
    578080:  ("PUBG",              "Battle Royale"),
    1172620: ("Sea of Thieves",    "Adventure"),
    1091500: ("Cyberpunk 2077",    "RPG"),
    945360:  ("Among Us",         "Social Deduction"),
    1422450: ("Vampire Survivors", "Rogue-like"),
}

app_meta_bc = sc.broadcast(app_metadata_local)

broadcast_joined_rdd = (
    reviews_rdd
    .filter(lambda r: r[F_APP_ID] in app_meta_bc.value)
    .map(lambda r: (
        r[F_APP_ID],
        app_meta_bc.value[r[F_APP_ID]][0],   # app_name
        app_meta_bc.value[r[F_APP_ID]][1],   # genre
        r[F_REVIEW_ID],
        r[F_RECOMMENDED],
        r[F_WEIGHTED_VOTE_SCORE],
        r[F_PLAYTIME_FOREVER],
    ))
)

with timer("Q8 — Broadcast Join (reviews x app metadata)"):
    for row in broadcast_joined_rdd.take(10):
        print(f"  app_id={row[0]}  app_name={row[1]}  genre={row[2]}  "
              f"review_id={row[3]}  rec={row[4]}  wvs={row[5]}  play={row[6]}")


# =============================================================================
# QUERY 9 — PARTITION PRUNING  (filter before aggregation)
# Goal: Analyse monthly review volumes and average scores for 2022–2023.
#
# RDD cost: there is no partition pruning in the RDD API. Even though we call
# filter(review_year in (2022, 2023)) early, Spark must read and deserialise
# every Parquet row-group before the Python lambda can discard non-target years.
# The DataFrame version, when reading a partitionBy("review_year") dataset,
# opens only the 2022/ and 2023/ directories — all other years are never read.
# =============================================================================

pruned_rdd = (
    reviews_rdd
    .filter(lambda r: r[F_REVIEW_YEAR] in (2022, 2023))
    .map(lambda r: (
        (r[F_REVIEW_YEAR], r[F_REVIEW_MONTH]),
        (r[F_WEIGHTED_VOTE_SCORE] or 0.0, 1),
    ))
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    .map(lambda kv: (
        kv[0][0],                             # review_year
        kv[0][1],                             # review_month
        kv[1][1],                             # monthly_reviews
        round(kv[1][0] / kv[1][1], 4),        # avg_score
    ))
    .sortBy(lambda r: (r[0], r[1]))
)

with timer("Q9 — Partition Pruning (2022–2023 monthly summary)"):
    for row in pruned_rdd.take(24):
        print(f"  year={row[0]}  month={row[1]}  "
              f"monthly_reviews={row[2]}  avg_score={row[3]}")


# =============================================================================
# QUERY 10 — SORT-MERGE JOIN
# Goal: Join reviews with user_profiles on author_steamid.
#
# RDD cost: .join() on two RDDs always performs a full shuffle-sort (Exchange
# hashpartitioning on both sides) followed by a merge pass — identical
# conceptually to SortMergeJoin in the DataFrame API. However, the DataFrame
# pre-repartition optimisation (letting AQE detect matching partitioning and
# skip one Exchange) is not available here. Both sides always shuffle.
# The is_veteran_account flag is a plain Python boolean — no Column expression.
# =============================================================================

user_profiles_local = [
    (76561198000000001, "NightStrike",  1200, 315),
    (76561198000000002, "VoidWalker",   890,  42),
    (76561198000000003, "CatalystKing", 2100, 178),
    (76561198000000004, "LagSlayer",    450,  9),
    (76561198000000005, "FrostByte",    3300, 501),
    (76561198000000006, "GankMaster",   760,  88),
    (76561198000000007, "SparkArcher",  1980, 234),
    (76561198000000008, "ManaVault",    550,  61),
]

user_profiles_rdd = sc.parallelize(user_profiles_local)

# Key both sides by author_steamid.
reviews_keyed      = reviews_rdd.map(lambda r: (r[F_AUTHOR_STEAMID], r))
user_profiles_keyed = user_profiles_rdd.map(lambda u: (u[0], u))

sort_merge_joined_rdd = (
    reviews_keyed
    .join(user_profiles_keyed)   # (author_steamid, (review_row, user_row))
    .map(lambda kv: (
        kv[0],                       # author_steamid
        kv[1][1][1],                 # username
        kv[1][1][2],                 # account_age_days
        kv[1][1][3],                 # total_reviews
        kv[1][0][F_APP_ID],          # app_id
        kv[1][0][F_REVIEW_ID],       # review_id
        kv[1][0][F_RECOMMENDED],     # recommended
        kv[1][0][F_WEIGHTED_VOTE_SCORE],  # weighted_vote_score
        kv[1][0][F_PLAYTIME_FOREVER],     # author_playtime_forever
        kv[1][1][2] >= 365,          # is_veteran_account
    ))
)

with timer("Q10 — Sort-Merge Join (reviews x user profiles)"):
    for row in sort_merge_joined_rdd.take(15):
        print(f"  steamid={row[0]}  user={row[1]}  age={row[2]}  "
              f"app_id={row[4]}  rec={row[6]}  wvs={row[7]}  veteran={row[9]}")


# =============================================================================
# CLEANUP
# =============================================================================

elite_rdd.unpersist()
app_meta_bc.destroy()

spark.stop()
