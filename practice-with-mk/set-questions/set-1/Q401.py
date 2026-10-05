from typing import List, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType
)

from pyspark.sql.functions import (
    col,
    to_date,
    datediff,
    when,
    sum,
    avg,
    count
)


# ============================================================
# 1. DEFINE SCHEMA
# ============================================================

def define_schema() -> StructType:

    return StructType([
        StructField("txn_id", StringType(), True),
        StructField("order_date", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("region", StringType(), True),
        StructField("channel", StringType(), True),
        StructField("category", StringType(), True),
        StructField("product_id", StringType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("unit_price", DoubleType(), True),
        StructField("discount_rate", DoubleType(), True),
        StructField("returned", StringType(), True),
        StructField("ship_date", StringType(), True),
        StructField("delivery_date", StringType(), True)
    ])


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data(
    spark: SparkSession,
    path: str,
    schema: StructType
) -> DataFrame:

    return spark.read.csv(
        path,
        header=True,
        schema=schema
    )


# ============================================================
# 3. PARSE DATES
# ============================================================

def parse_dates(df: DataFrame) -> DataFrame:

    return (
        df
        .withColumn("order_date", to_date("order_date"))
        .withColumn("ship_date", to_date("ship_date"))
        .withColumn("delivery_date", to_date("delivery_date"))
    )


# ============================================================
# 4. ADD GROSS AMOUNT
# ============================================================

def add_gross_amount(df: DataFrame) -> DataFrame:

    return df.withColumn(
        "gross_amount",
        col("quantity") * col("unit_price")
    )


# ============================================================
# 5. ADD NET AMOUNT
# ============================================================

def add_net_amount(df: DataFrame) -> DataFrame:

    df = add_gross_amount(df)

    return df.withColumn(
        "net_amount",
        col("gross_amount") * (1 - col("discount_rate"))
    )


# ============================================================
# 6. ADD DELIVERY DAYS
# ============================================================

def add_delivery_days(df: DataFrame) -> DataFrame:

    return df.withColumn(
        "delivery_days",
        datediff(
            col("delivery_date"),
            col("ship_date")
        )
    )


# ============================================================
# 7. FLAG ON-TIME DELIVERY
# ============================================================

def flag_on_time_delivery(
    df: DataFrame,
    max_days: int
) -> DataFrame:

    df = add_delivery_days(df)

    return df.withColumn(
        "is_on_time",
        when(
            col("delivery_days") <= max_days,
            True
        ).otherwise(False)
    )


# ============================================================
# 8. FILTER RETURNED ORDERS
# ============================================================

def filter_returned_orders(df: DataFrame) -> DataFrame:

    return df.filter(
        col("returned") == "Y"
    )


# ============================================================
# 9. FILTER BY REGION
# ============================================================

def filter_by_region(
    df: DataFrame,
    region: str
) -> DataFrame:

    return df.filter(
        col("region") == region
    )


# ============================================================
# 10. TOP N CUSTOMERS BY SPEND
# ============================================================

def top_n_customers_by_spend(
    df: DataFrame,
    n: int
) -> DataFrame:

    df = add_net_amount(df)

    return (
        df
        .groupBy("customer_id")
        .agg(
            sum("net_amount").alias("total_spend")
        )
        .orderBy(
            col("total_spend").desc(),
            col("customer_id").asc()
        )
        .limit(n)
    )


# ============================================================
# 11. REVENUE BY CATEGORY
# ============================================================

def revenue_by_category(
    df: DataFrame
) -> DataFrame:

    df = add_net_amount(df)

    return (
        df
        .groupBy("category")
        .agg(
            sum("net_amount").alias("total_revenue")
        )
    )


# ============================================================
# 12. TOP CATEGORY BY REVENUE
# ============================================================

def top_category_by_revenue(
    df: DataFrame
) -> str:

    df = add_net_amount(df)

    row = (
        df
        .groupBy("category")
        .agg(
            sum("net_amount").alias("total_revenue")
        )
        .orderBy(
            col("total_revenue").desc(),
            col("category").asc()
        )
        .first()
    )

    if row:
        return row["category"]

    return None


# ============================================================
# 13. AVERAGE DISCOUNT BY CHANNEL
# ============================================================

def avg_discount_by_channel(
    df: DataFrame
) -> DataFrame:

    return (
        df
        .groupBy("channel")
        .agg(
            avg("discount_rate").alias("avg_discount")
        )
    )


# ============================================================
# 14. DAILY REVENUE TREND
# ============================================================

def daily_revenue_trend(
    df: DataFrame
) -> DataFrame:

    df = add_net_amount(df)

    return (
        df
        .groupBy("order_date")
        .agg(
            sum("net_amount").alias("daily_revenue")
        )
        .orderBy("order_date")
    )


# ============================================================
# 15. HIGH VALUE ORDERS
# ============================================================

def high_value_orders(
    df: DataFrame,
    threshold: float
) -> DataFrame:

    df = add_net_amount(df)

    return df.filter(
        col("net_amount") > threshold
    )


# ============================================================
# 16. COUNT LATE DELIVERIES
# ============================================================

def count_late_deliveries(
    df: DataFrame,
    max_days: int
) -> int:

    df = add_delivery_days(df)

    return (
        df
        .filter(
            col("delivery_days") > max_days
        )
        .count()
    )


# ============================================================
# 17. RETURN RATE BY CATEGORY
# ============================================================

def return_rate_by_category(
    df: DataFrame
) -> DataFrame:

    df = (
        df
        .groupBy("category")
        .agg(
            count("*").alias("total_cnt"),
            sum(
                when(
                    col("returned") == "Y",
                    1
                ).otherwise(0)
            ).alias("returned_cnt")
        )
    )

    return df.withColumn(
        "return_rate",
        col("returned_cnt") / col("total_cnt")
    )


# ============================================================
# 18. BEST SELLING PRODUCT
# ============================================================

def best_selling_product(
    df: DataFrame
) -> Tuple[str, int]:

    row = (
        df
        .groupBy("product_id")
        .agg(
            sum("quantity").alias("total_qty")
        )
        .orderBy(
            col("total_qty").desc(),
            col("product_id").asc()
        )
        .first()
    )

    if row:
        return (
            row["product_id"],
            row["total_qty"]
        )

    return None


# ============================================================
# 19. LIST CHANNELS
# ============================================================

def list_channels(
    df: DataFrame
) -> List[str]:

    df = (
        df
        .select("channel")
        .distinct()
        .orderBy(
            col("channel").asc()
        )
    )

    rows = df.collect()

    result = []

    for row in rows:
        result.append(
            row["channel"]
        )

    return result


# ============================================================
# 20. ORDERS IN DATE RANGE
# ============================================================

def orders_in_date_range(
    df: DataFrame,
    start_date: str,
    end_date: str
) -> DataFrame:

    return df.filter(
        (col("order_date") >= start_date) &
        (col("order_date") <= end_date)
    )
