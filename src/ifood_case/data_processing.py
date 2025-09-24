from typing import Tuple

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when


class DataProcessing:
    def __init__(self, offers: DataFrame, transactions: DataFrame, profile: DataFrame):
        self.offers = offers
        self.transactions = transactions
        self.profile = profile

    def transform(self) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        self.__restructure_transactions()
        self.__age_group()
        df_joined = self.__join_dfs()
        return df_joined, self.offers, self.transactions, self.profile

    def __restructure_transactions(self):
        self.transactions = self.transactions.select(
            col("account_id"),
            col("event"),
            col("time_since_test_start"),
            col("value.offer_id").alias("offer_completed"),
            col(
                r"value.offer id",
            ).alias("offer_received_viewed"),
            col("value.amount").alias("amount"),
            col("value.reward").alias("reward"),
        ).drop(col("value"))


    def __age_group(self):
        self.profile = self.profile.withColumn(
            "age_group",
            when((col("age") >= 18) & (col("age") <= 25), "18-25")
            .when((col("age") >= 26) & (col("age") <= 35), "26-35")
            .when((col("age") >= 36) & (col("age") <= 50), "36-50")
            .when(col("age") > 50, "51+"),
        )

    def __join_dfs(self) -> DataFrame:
        offers1 = self.offers.alias("offers1").withColumnRenamed("offer_type", "offer_type_1")
        offers2 = self.offers.alias("offers2").withColumnRenamed("offer_type", "offer_type_2")

        df_joined = self.transactions.join(self.profile, self.transactions.account_id == self.profile.id, "left")
        df_joined = df_joined.join(offers1, offers1.id == df_joined.offer_completed, "left")
        df_joined = df_joined.join(offers2, offers2.id == df_joined.offer_received_viewed, "left")

        return df_joined
