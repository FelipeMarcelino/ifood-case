"""This module provides the DataProcessing class, responsible for cleaning,
restructuring, and joining the raw iFood case study DataFrames.
"""
from typing import Tuple

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when


class DataProcessing:
    """Orchestrates the initial data processing and joining pipeline.

    This class takes the raw offers, transactions, and profile DataFrames,
    applies cleaning and transformation logic, and joins them into a
    single unified DataFrame.

    Attributes
    ----------
    offers : pyspark.sql.DataFrame
        The raw DataFrame containing offer details.
    transactions : pyspark.sql.DataFrame
        The raw DataFrame containing the log of all events.
    profile : pyspark.sql.DataFrame
        The raw DataFrame containing customer profile data.

    """

    def __init__(self, offers: DataFrame, transactions: DataFrame, profile: DataFrame) -> None:
        """Initializes the DataProcessing class.

        Parameters
        ----------
        offers : pyspark.sql.DataFrame
            The raw DataFrame containing offer details.
        transactions : pyspark.sql.DataFrame
            The raw DataFrame containing the log of all events.
        profile : pyspark.sql.DataFrame
            The raw DataFrame containing customer profile data.

        """
        self._offers = offers
        self._transactions = transactions
        self._profile = profile

    def transform(self) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """Executes the full data processing and joining pipeline.

        This method orchestrates the cleaning of profile data, restructuring of
        transaction data, and the final joining of all three source DataFrames.

        Returns
        -------
        Tuple[pyspark.sql.DataFrame, ...]
            A tuple containing four DataFrames:
            - The final joined DataFrame.
            - The original offers DataFrame.
            - The restructured transactions DataFrame.
            - The processed profile DataFrame with age groups.

        """
        transactions_processed = self._restructure_transactions(self._transactions)
        profile_processed = self._create_age_groups(self._profile)
        df_joined = self._join_dataframes(
            transactions_processed, profile_processed, self._offers,
        )
        return df_joined, self._offers, transactions_processed, profile_processed

    @staticmethod
    def _restructure_transactions(transactions_df: DataFrame) -> DataFrame:
        """Restructures the transactions DataFrame by flattening the nested 'value' column.

        Parameters
        ----------
        transactions_df : pyspark.sql.DataFrame
            The raw transactions DataFrame.

        Returns
        -------
        pyspark.sql.DataFrame
            A restructured DataFrame with flattened columns for offer IDs,
            amount, and reward.

        """
        # Select and rename fields from the nested 'value' struct.
        # The redundant .drop("value") is removed as .select() already handles this.
        return transactions_df.select(
            col("account_id"),
            col("event"),
            col("time_since_test_start"),
            col("value.offer_id").alias("offer_completed"),
            col("value.`offer id`").alias("offer_received_viewed"), # Handles space in name
            col("value.amount").alias("amount"),
            col("value.reward").alias("reward"),
        )

    @staticmethod
    def _create_age_groups(profile_df: DataFrame) -> DataFrame:
        """Adds an 'age_group' column to the profile DataFrame by binning the 'age'.

        Parameters
        ----------
        profile_df : pyspark.sql.DataFrame
            The profile DataFrame.

        Returns
        -------
        pyspark.sql.DataFrame
            The profile DataFrame with the new 'age_group' column.

        """
        return profile_df.withColumn(
            "age_group",
            when((col("age") >= 18) & (col("age") <= 25), "18-25")
            .when((col("age") >= 26) & (col("age") <= 35), "26-35")
            .when((col("age") >= 36) & (col("age") <= 50), "36-50")
            .when(col("age") > 50, "51+")
            .otherwise("Unknown"),  # BUG FIX: Handles ages under 18 and nulls
        )

    @staticmethod
    def _join_dataframes(
        transactions: DataFrame, profile: DataFrame, offers: DataFrame,
    ) -> DataFrame:
        """Joins the processed transactions, profile, and offers DataFrames.

        It handles joining the offers DataFrame twice by using aliases to avoid
        ambiguous column errors.

        Parameters
        ----------
        transactions : pyspark.sql.DataFrame
            The processed transactions DataFrame.
        profile : pyspark.sql.DataFrame
            The processed profile DataFrame.
        offers : pyspark.sql.DataFrame
            The original offers DataFrame.

        Returns
        -------
        pyspark.sql.DataFrame
            A single, wide DataFrame containing all joined information.

        """
        # Alias the offers DataFrame to join it twice for the different offer ID types
        offers1 = offers.alias("offers1").withColumnRenamed("offer_type", "offer_type_1")
        offers2 = offers.alias("offers2").withColumnRenamed("offer_type", "offer_type_2")

        # Join transactions with profiles
        df_joined = transactions.join(
            profile, transactions.account_id == profile.id, "left",
        )
        # Join with the first offers alias on the 'offer_completed' key
        df_joined = df_joined.join(
            offers1, offers1.id == df_joined.offer_completed, "left",
        )
        # Join with the second offers alias on the 'offer_received_viewed' key
        df_joined = df_joined.join(
            offers2, offers2.id == df_joined.offer_received_viewed, "left",
        )

        return df_joined
