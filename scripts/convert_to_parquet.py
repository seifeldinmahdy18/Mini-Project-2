import os
os.environ["JAVA_HOME"] = "/usr/local/sdkman/candidates/java/21.0.10-ms"
os.environ["JAVA_TOOL_OPTIONS"] = (
    "-Djava.security.manager=allow "
    "--add-opens=java.base/javax.security.auth=ALL-UNNAMED"
)
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

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

CSV_PATH     = "data/reviews-1-115.csv"
PARQUET_PATH = "data/steam_reviews.parquet"

# Read with header only — no schema — so columns are matched by name, not
# position. This is safe when review text contains commas/newlines.
raw = (
    spark.read
    .option("header",    "true")
    .option("multiLine", "true")
    .option("escape",    '"')
    .csv(CSV_PATH)
)

# Cast each column to its correct type.
# votes_funny uses LongType: the dataset contains 4294967295 (UINT32_MAX)
# which overflows INT. All other counts fit in INT.
# try_cast (F.try_cast) silently returns NULL for any malformed stragglers
# instead of aborting the job.
def try_cast(col_name, dtype):
    return F.expr(f"try_cast(`{col_name}` AS {dtype})").alias(col_name)

df = (
    raw
    .withColumn("steamid",                try_cast("steamid",                "BIGINT"))
    .withColumn("appid",                  try_cast("appid",                  "INT"))
    .withColumn("voted_up",               F.expr("try_cast(lower(voted_up) AS BOOLEAN)").alias("voted_up"))
    .withColumn("votes_up",               try_cast("votes_up",               "INT"))
    .withColumn("votes_funny",            try_cast("votes_funny",            "BIGINT"))
    .withColumn("weighted_vote_score",    try_cast("weighted_vote_score",    "FLOAT"))
    .withColumn("playtime_forever",       try_cast("playtime_forever",       "BIGINT"))
    .withColumn("playtime_at_review",     try_cast("playtime_at_review",     "BIGINT"))
    .withColumn("num_games_owned",        try_cast("num_games_owned",        "INT"))
    .withColumn("num_reviews",            try_cast("num_reviews",            "INT"))
    .withColumn("unix_timestamp_created", try_cast("unix_timestamp_created", "BIGINT"))
    .withColumn("unix_timestamp_updated", try_cast("unix_timestamp_updated", "BIGINT"))
    .withColumn("review_year",            F.year(F.to_timestamp("unix_timestamp_created")))
    .withColumn("review_month",           F.month(F.to_timestamp("unix_timestamp_created")))
)

df.write.mode("overwrite").parquet(PARQUET_PATH)

print(f"Done. Parquet written to {PARQUET_PATH}")

spark.stop()
