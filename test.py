from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# -------------------------------------------------
# Create Spark session (mobile-safe config)
# -------------------------------------
spark = (
    SparkSession.builder
    .appName("Termux-PySpark-Sample")
    .master("local[2]")                 # limit cores
    .config("spark.driver.memory", "1g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

# ---------------------------------------
# ---------------------------------------

dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]

rows = []

# 20 RFs
for d in dates:
    for i in range(1, 21):
        rows.append((d, f"RF{i}", float(i) * 0.01))

# 3 DVs
for d in dates:
    rows.extend([
        (d, "DV1", 0.5),
        (d, "DV2", 0.3),
        (d, "DV3", 0.7),
    ])

df_transform_clean = spark.createDataFrame(
    rows,
    ["time_series_date", "shock_rule_id", "return_value"]
)
df_transform_clean.show()
df_shock_clean = spark.createDataFrame(
    [
        # DVs
        ("DV1", 1, False, ["RF3", "RF4"]),
        ("DV2", 1, True,  ["RF6"]),
        ("DV3", 1, True,  []),

        # RFs
        *[(f"RF{i}", 0, False, []) for i in range(1, 21)]
    ],
    ["shock_rule_id", "is_mlr", "preferred", "nonPreferred"]
)

df_shock_clean.show()
NUM_VARIABLES = 3   # total RFs per DV

dv_df  = df_shock_clean.filter(F.col("is_mlr") == 1)
idv_df = df_shock_clean.filter(F.col("is_mlr") == 0)
ts_df  = df_transform_clean

results = []

for dv_row in dv_df.collect():

    dv = dv_row["shock_rule_id"]
    preferred_flag = dv_row["preferred"]
    non_preferred = dv_row["nonPreferred"] or []

    # -------------------------
    # Initialize residual = DV
    # -------------------------
    residual = (
        ts_df
        .filter(F.col("shock_rule_id") == dv)
        .select(
            "time_series_date",
            F.col("return_value").alias("residual")
        )
    )

    # -------------------------
    # Candidate RFs
    # -------------------------
    idv_candidates = (
        ts_df
        .join(idv_df.select("shock_rule_id"), "shock_rule_id")
        .filter(~F.col("shock_rule_id").isin(non_preferred))
    )

    selected = []

    # -------------------------
    # Mandatory preferred RFs
    # -------------------------
    if preferred_flag:

        pref_vars = (
            idv_df
            .filter(F.col("preferred") == True)
            .select("shock_rule_id")
            .rdd.flatMap(lambda x: x)
            .collect()
        )

        for pv in pref_vars:
            beta = (
                residual.alias("r")
                .join(
                    ts_df.filter(F.col("shock_rule_id") == pv).alias("x"),
                    "time_series_date"
                )
                .select(
                    F.sum(F.col("r.residual") * F.col("x.return_value")) /
                    F.sum(F.col("x.return_value") ** 2)
                )
                .collect()[0][0]
            )

            residual = (
                residual.alias("r")
                .join(
                    ts_df.filter(F.col("shock_rule_id") == pv).alias("x"),
                    "time_series_date"
                )
                .select(
                    "time_series_date",
                    (F.col("r.residual") - beta * F.col("x.return_value")).alias("residual")
                )
            )

            selected.append(pv)

    # -------------------------
    # Greedy forward selection
    # -------------------------
    while len(selected) < NUM_VARIABLES:

        corr_stats = (
            residual.alias("r")
            .join(idv_candidates.alias("x"), "time_series_date")
            .filter(~F.col("x.shock_rule_id").isin(selected))
            .groupBy("x.shock_rule_id")
            .agg(
                F.sum(F.col("r.residual") * F.col("x.return_value")).alias("sum_rx"),
                F.sum(F.col("r.residual") ** 2).alias("sum_r2"),
                F.sum(F.col("x.return_value") ** 2).alias("sum_x2")
            )
            .withColumn(
                "corr",
                F.abs(F.col("sum_rx")) /
                F.sqrt(F.col("sum_r2") * F.col("sum_x2"))
            )
        )

        best = corr_stats.orderBy(F.desc("corr")).limit(1).collect()
        if not best:
            break

        best_rf = best[0]["shock_rule_id"]
        beta = best[0]["sum_rx"] / best[0]["sum_x2"]

        residual = (
            residual.alias("r")
            .join(
                ts_df.filter(F.col("shock_rule_id") == best_rf).alias("x"),
                "time_series_date"
            )
            .select(
                "time_series_date",
                (F.col("r.residual") - beta * F.col("x.return_value")).alias("residual")
            )
        )

        selected.append(best_rf)

    results.append((dv, selected))

final_df = spark.createDataFrame(
    results,
    ["dv", "idv_final"]
)

final_df.show(truncate=False)

# -------------------------------------------------
# Stop Spark
# -------------------------------------------------
spark.stop()
