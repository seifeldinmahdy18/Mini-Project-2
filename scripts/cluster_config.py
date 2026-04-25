"""
Spark Cluster Configuration — Steam Reviews Project
====================================================

WHAT CLUSTER CONFIGURATION MEANS IN SPARK
------------------------------------------
Spark "cluster configuration" controls how memory, CPU, and network resources
are allocated across the driver and executors. In local mode (used here) the
driver and executor run in the same JVM process, so the settings still determine
how much heap is reserved for execution vs. storage, and how much parallelism
Spark exposes to the task scheduler.

DRIVER MEMORY vs. EXECUTOR MEMORY
-----------------------------------
- spark.driver.memory   : heap for the driver JVM — collects results, runs
                          the query planner, holds broadcast variables.
- spark.executor.memory : heap per executor JVM — does the actual data
                          processing (map, shuffle, aggregate).
  In local mode both settings apply to the single JVM that Spark starts.

WHY SHUFFLE PARTITIONS MATTER
------------------------------
Every GROUP BY, JOIN, or DISTINCT triggers a shuffle. Spark breaks the shuffled
data into spark.sql.shuffle.partitions pieces. Too few → large partitions,
memory pressure, spill to disk. Too many → thousands of tiny tasks, scheduler
overhead dominates. A good rule of thumb: target ~128 MB per partition.

HOW AQE INTERACTS WITH PARTITION CONFIG
-----------------------------------------
Adaptive Query Execution (AQE) re-optimises the plan at runtime after each
shuffle stage completes:
  - coalescePartitions : if actual shuffle output is small, AQE merges
    neighbouring partitions so you don't spin up 200 tasks for 10 MB of data.
  - skewJoin           : detects partitions that are much larger than the
    median and splits them across multiple tasks automatically.
AQE makes the initial spark.sql.shuffle.partitions a ceiling rather than a
fixed cost — setting it high (400) is safe because AQE will coalesce down.

LOCAL[*] vs LOCAL[2] vs LOCAL[1]
----------------------------------
  local[1]  — single-threaded; one task runs at a time. Good for debugging
              because stack traces are deterministic.
  local[2]  — two parallel tasks; matches a 2-vCPU Codespace and avoids
              over-subscribing the machine.
  local[*]  — one thread per logical CPU core on the host. Maximum throughput
              on a beefy machine; can cause contention on shared environments.
"""

import os
import time

os.environ["JAVA_HOME"] = "/usr/local/sdkman/candidates/java/21.0.10-ms"
os.environ["JAVA_TOOL_OPTIONS"] = (
    "-Djava.security.manager=allow "
    "--add-opens=java.base/javax.security.auth=ALL-UNNAMED"
)

from pyspark.sql import SparkSession

DATA_PATH = "data/steam_reviews.parquet"

SHARED_CONFIGS = {
    "spark.sql.adaptive.enabled":                    "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled":           "true",
    "spark.serializer":                              "org.apache.spark.serializer.KryoSerializer",
    "spark.kryoserializer.buffer.max":               "512m",
    "spark.sql.parquet.filterPushdown":              "true",
    "spark.sql.parquet.mergeSchema":                 "false",
    "spark.sql.autoBroadcastJoinThreshold":          "52428800",
    "spark.memory.fraction":                         "0.8",
    "spark.memory.storageFraction":                  "0.3",
}

PROFILES = {
    "default": {
        "spark.master":                  "local[2]",
        "spark.driver.memory":           "4g",
        "spark.executor.memory":         "2g",
        "spark.sql.shuffle.partitions":  "200",
    },
    "low": {
        "spark.master":                  "local[1]",
        "spark.driver.memory":           "2g",
        "spark.executor.memory":         "1g",
        "spark.sql.shuffle.partitions":  "50",
    },
    "high": {
        "spark.master":                  "local[*]",
        "spark.driver.memory":           "6g",
        "spark.executor.memory":         "3g",
        "spark.sql.shuffle.partitions":  "400",
    },
}

ALL_TRACKED_KEYS = list(PROFILES["default"].keys()) + list(SHARED_CONFIGS.keys())


def get_configured_session(app_name: str, profile: str = "default") -> SparkSession:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Choose from: {list(PROFILES)}")

    profile_configs = {**PROFILES[profile], **SHARED_CONFIGS}

    builder = SparkSession.builder.appName(app_name)
    for key, value in profile_configs.items():
        if key == "spark.master":
            builder = builder.master(value)
        else:
            builder = builder.config(key, value)

    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("WARN")
    return session


def _print_config_table(spark: SparkSession, profile: str) -> None:
    conf = spark.sparkContext.getConf()
    col_w = 46

    print(f"\n  {'Setting':<{col_w}} Value")
    print(f"  {'-' * col_w} {'─' * 20}")
    for key in ALL_TRACKED_KEYS:
        value = conf.get(key, "(not set)")
        print(f"  {key:<{col_w}} {value}")


if __name__ == "__main__":
    for profile_name in ("default", "low", "high"):
        print(f"\n{'=' * 65}")
        print(f"  PROFILE: {profile_name.upper()}")
        print(f"{'=' * 65}")

        spark = get_configured_session("SteamReviews-ClusterConfig", profile=profile_name)

        _print_config_table(spark, profile_name)

        print(f"\n  Reading: {DATA_PATH}")
        t0 = time.perf_counter()
        count = spark.read.parquet(DATA_PATH).count()
        elapsed = time.perf_counter() - t0

        print(f"  Row count : {count:,}")
        print(f"  Time      : {elapsed:.3f}s")

        spark.stop()

    print(f"\n{'=' * 65}")
    print("  All profiles complete.")
    print(f"{'=' * 65}\n")
