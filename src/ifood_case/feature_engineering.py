"""This module contains the FeatureEngineering class, responsible for transforming
raw iFood case study data into a feature-rich dataset for modeling.
"""

import math

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from ifood_case.utils import get_column_types


class FeatureEngineering:
    """Orchestrates the feature engineering pipeline for the iFood case study.

    This class takes raw PySpark DataFrames for offers, transactions, and
    profiles, and applies a series of transformations to clean the data,
    create new features, and produce a single, unified DataFrame ready for
    a machine learning model.
    """

    def __init__(self, offers: DataFrame, transactions: DataFrame, profile: DataFrame):
        """Initializes the FeatureEngineering class with the raw dataframes.

        Parameters
        ----------
        offers : pyspark.sql.DataFrame
            DataFrame containing offer details.
        transactions : pyspark.sql.DataFrame
            DataFrame containing the log of all transactions and offer events.
            It's assumed this DataFrame has a unified 'offer_id' column.
        profile : pyspark.sql.DataFrame
            DataFrame containing customer profile data.

        """
        self.offers = offers
        self.transactions = transactions
        self.profile = profile

    def transform(self) -> tuple[DataFrame, list[str], list[str]]:
        """Executes the full feature engineering pipeline.

        This method orchestrates the creation of all features and joins them
        into a single, final DataFrame.

        Returns
        -------
        tuple[pyspark.sql.DataFrame, list[str], list[str]]
            A tuple containing:
            - The final feature-engineered DataFrame.
            - A list of numerical column names.
            - A list of categorical column names.

        """
        target_df = self._create_target_variable()
        point_in_time_features_df = self._create_point_in_time_features()
        last_offer_features_df = self._create_last_offer_viewed_feature()
        profile_features_df = self._create_profile_features()
        offer_features_df = self._create_static_offer_features()

        # Join 1: Base (target) with point-in-time customer features
        final_df = (
            target_df.alias("t")
            .join(
                point_in_time_features_df.alias("p"),
                on=[
                    F.col("t.offer_id") == F.col("p.offer_id_received"),
                    F.col("t.account_id") == F.col("p.account_id"),
                    F.col("t.time_received") == F.col("p.time_received"),
                ],
                how="left",
            )
            .drop(F.col("p.account_id"), F.col("p.offer_id_received"), F.col("p.time_received"))
        )

        # Join 2: Add the last offer viewed feature
        final_df = (
            final_df.alias("f")
            .join(
                last_offer_features_df.alias("l"),
                on=[
                    F.col("f.offer_id") == F.col("l.offer_id_received"),
                    F.col("f.account_id") == F.col("l.account_id"),
                    F.col("f.time_received") == F.col("l.time_received"),
                ],
                how="left",
            )
            .drop(F.col("l.time_received"), F.col("l.account_id"), F.col("l.offer_id_received"))
        )

        # Join 3: Add customer profile features
        final_df = (
            final_df.alias("f")
            .join(
                profile_features_df.alias("p"),
                on=(F.col("f.account_id") == F.col("p.id")),
                how="left",
            )
            .drop(F.col("p.id"))
        )

        # Join 4: Add static offer features
        final_df = (
            final_df.alias("f")
            .join(
                offer_features_df.alias("o"),
                on=(F.col("f.offer_id") == F.col("o.id")),
                how="left",
            )
            .drop(F.col("o.id"))
        )

        numerical_columns, categorical_columns = get_column_types(final_df)

        return final_df, numerical_columns, categorical_columns

    def _create_target_variable(self) -> DataFrame:
        """Creates the target variable for the modeling dataset based on offer outcomes.

        It defines a success event (target=1) for BOGO/discount offers if they
        are completed, and for informational offers if they are followed by a
        transaction within their duration.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame with the grain (account_id, offer_id, time_received)
            and the binary `target` column.

        """
        bogo_discount_success = (
            self.transactions.filter(F.col("event") == "offer completed")
            .select("account_id", F.col("offer_completed").alias("successful_offer_id"))
            .distinct()
        )

        informational_offer_ids = [
            row.id for row in self.offers.filter(F.col("offer_type") == "informational").select("id").collect()
        ]

        info_views = (
            self.transactions.filter(
                (F.col("event") == "offer viewed") & (F.col("offer_received_viewed").isin(informational_offer_ids)),
            )
            .alias("v")
            .join(self.offers.alias("o"), F.col("v.offer_received_viewed") == F.col("o.id"))
            .select(
                F.col("v.account_id"),
                F.col("v.offer_received_viewed"),
                F.col("v.time_since_test_start").alias("view_time"),
                F.col("o.duration"),
            )
        )

        all_transactions = self.transactions.filter(F.col("event") == "transaction").alias("t")

        informational_success = (
            info_views.join(
                all_transactions,
                info_views.account_id == all_transactions.account_id,
                "inner",
            )
            .filter(
                (F.col("t.time_since_test_start") >= F.col("view_time"))
                & (F.col("t.time_since_test_start") <= F.col("view_time") + F.col("duration")),
            )
            .select(
                F.col("v.account_id"),
                F.col("v.offer_received_viewed").alias("successful_offer_id"),
            )
            .distinct()
        )

        all_successful_offers = bogo_discount_success.unionByName(informational_success)

        offers_received_base = (
            self.transactions.filter(F.col("event") == "offer received")
            .select(
                "account_id",
                F.col("offer_received_viewed").alias("offer_id"),
                F.col("time_since_test_start").alias("time_received"),
            )
            .distinct()
        )

        target_dataset = (
            offers_received_base.alias("base")
            .join(
                all_successful_offers.alias("success"),
                (F.col("base.account_id") == F.col("success.account_id"))
                & (F.col("base.offer_id") == F.col("success.successful_offer_id")),
                "left",
            )
            .withColumn(
                "target",
                F.when(F.col("success.successful_offer_id").isNotNull(), 1).otherwise(0),
            )
            .select("base.account_id", "base.offer_id", "base.time_received", "target")
        )
        return target_dataset

    def _create_point_in_time_features(self) -> DataFrame:
        """Creates point-in-time features for each received offer event.

        For each 'offer received' event, features like cumulative spend and
        historical conversion rate are calculated using only data available
        *before* that event, preventing data leakage.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame with the grain (account_id, offer_id_received, time_received)
            and various point-in-time features.

        """
        window_spec = Window.partitionBy("account_id").orderBy("time_since_test_start")

        features_over_time = (
            self.transactions.withColumn(
                "amount_if_transaction",
                F.when(F.col("event") == "transaction", F.col("amount")).otherwise(None),
            )
            .withColumn("spend_cumulative", F.sum("amount_if_transaction").over(window_spec))
            .withColumn(
                "transactions_cumulative_count",
                F.sum(F.when(F.col("event") == "transaction", 1).otherwise(0)).over(window_spec),
            )
            .withColumn("avg_ticket_cumulative", F.avg("amount_if_transaction").over(window_spec))
            .withColumn("max_ticket_cumulative", F.max("amount_if_transaction").over(window_spec))
            .withColumn("min_ticket_cumulative", F.min("amount_if_transaction").over(window_spec))
            .withColumn(
                "offers_viewed_cumulative_count",
                F.sum(F.when(F.col("event") == "offer viewed", 1).otherwise(0)).over(window_spec),
            )
            .withColumn(
                "offers_completed_cumulative_count",
                F.sum(F.when(F.col("event") == "offer completed", 1).otherwise(0)).over(window_spec),
            )
        )

        features_before_event = (
            features_over_time.withColumn("total_spend_before", F.lag("spend_cumulative", 1, 0).over(window_spec))
            .withColumn("transaction_count_before", F.lag("transactions_cumulative_count", 1, 0).over(window_spec))
            .withColumn("avg_ticket_before", F.lag("avg_ticket_cumulative", 1, 0).over(window_spec))
            .withColumn("max_ticket_before", F.lag("max_ticket_cumulative", 1, 0).over(window_spec))
            .withColumn("min_ticket_before", F.lag("min_ticket_cumulative", 1, 0).over(window_spec))
            .withColumn("offers_viewed_count_before", F.lag("offers_viewed_cumulative_count", 1, 0).over(window_spec))
            .withColumn(
                "offers_completed_count_before", F.lag("offers_completed_cumulative_count", 1, 0).over(window_spec),
            )
        )

        features_with_conv_rate = features_before_event.withColumn(
            "customer_conversion_rate_before",
            F.when(
                F.col("offers_viewed_count_before") > 0,
                (F.col("offers_completed_count_before") / F.col("offers_viewed_count_before")) * 100,
            ).otherwise(0.0),
        )

        final_features_for_offers = features_with_conv_rate.filter(F.col("event") == "offer received").select(
            "account_id",
            F.col("offer_received_viewed").alias("offer_id_received"),
            F.col("time_since_test_start").alias("time_received"),
            "total_spend_before",
            "transaction_count_before",
            "avg_ticket_before",
            "max_ticket_before",
            "min_ticket_before",
            "offers_viewed_count_before",
            "offers_completed_count_before",
            "customer_conversion_rate_before",
        )
        return final_features_for_offers

    def _create_last_offer_viewed_feature(self) -> DataFrame:
        """For each received offer, finds the type of the last offer viewed by
        the customer before that specific event.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame with the grain (account_id, offer_id_received, time_received)
            and the `last_offer_viewed_type` feature.

        """
        opportunities = self.transactions.filter(F.col("event") == "offer received").select(
            "account_id",
            F.col("offer_received_viewed").alias("offer_id_received"),
            F.col("time_since_test_start").alias("time_received"),
        )

        viewed_events = self.transactions.filter(F.col("event") == "offer viewed").select(
            F.col("account_id"),
            F.col("offer_received_viewed").alias("offer_id_viewed"),
            F.col("time_since_test_start").alias("time_viewed"),
        )

        joined_df = opportunities.alias("opp").join(
            viewed_events.alias("v"),
            (F.col("opp.account_id") == F.col("v.account_id")) & (F.col("opp.time_received") > F.col("v.time_viewed")),
            "left",
        )

        window_spec = Window.partitionBy("opp.account_id", "opp.offer_id_received", "opp.time_received").orderBy(
            F.col("v.time_viewed").desc(),
        )
        ranked_views = joined_df.withColumn("rank", F.row_number().over(window_spec))

        last_viewed_offer_id_df = ranked_views.filter(F.col("rank") == 1).select(
            "opp.account_id",
            "opp.offer_id_received",
            "opp.time_received",
            F.col("v.offer_id_viewed").alias("last_offer_viewed_id"),
        )

        last_viewed_offer_type_df = (
            last_viewed_offer_id_df.join(
                self.offers.alias("o_details"),
                last_viewed_offer_id_df.last_offer_viewed_id == F.col("o_details.id"),
                "left",
            )
            .select(
                "account_id",
                "offer_id_received",
                "time_received",
                F.col("o_details.offer_type").alias("last_offer_viewed_type"),
            )
            .na.fill({"last_offer_viewed_type": "None"})
        )

        return last_viewed_offer_type_df

    def _clean_profile(self, profile_df: DataFrame) -> DataFrame:
        """Performs initial cleaning on the profile DataFrame.

        Replaces age outlier (118) with null.

        Parameters
        ----------
        profile_df : pyspark.sql.DataFrame
            The raw profile DataFrame.

        Returns
        -------
        pyspark.sql.DataFrame
            The cleaned profile DataFrame.

        """
        return profile_df.withColumn(
            "age",
            F.when(F.col("age") == 118, None).otherwise(F.col("age")),
        )

    def _create_profile_features(self) -> DataFrame:
        """Creates all features derived from the customer profile data.

        This method orchestrates the cleaning of the profile data and the
        creation of new features like age groups and cyclical date features.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame of profile features, keyed by customer `id`.

        """
        cleaned_profile = self._clean_profile(self.profile)
        profile_with_cyclical = self._create_cyclical_features(cleaned_profile)

        selected_profile_features = profile_with_cyclical.select(
            F.col("id"),
            F.col("gender"),
            F.col("age"),
            F.col("credit_card_limit"),
            F.col("month_sin"),
            F.col("month_cos"),
            F.col("dayofweek_sin"),
            F.col("dayofweek_cos"),
        )
        return selected_profile_features

    def _create_static_offer_features(self) -> DataFrame:
        """Creates static, one-hot encoded features and ratio features for each offer.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame of offer features, keyed by offer `id`.

        """
        # One-hot encode channels
        channels_df = self.offers.select(F.explode("channels").alias("channel"))
        distinct_channels = [row.channel for row in channels_df.distinct().collect()]

        offers_df = self.offers
        for channel in distinct_channels:
            offers_df = offers_df.withColumn(
                f"channel_is_{channel}",
                F.when(F.array_contains(F.col("channels"), channel), 1).otherwise(0),
            )

        # Create ratio feature
        offers_df = offers_df.withColumn(
            "reward_ratio",
            F.when(F.col("min_value") > 0, F.col("discount_value") / F.col("min_value")).otherwise(0),
        )

        # Select the final features
        feature_columns = [
            "id",
            "offer_type",
            "duration",
            "min_value",
            "discount_value",
            "reward_ratio",
        ] + [f"channel_is_{channel}" for channel in distinct_channels]

        return offers_df.select(*feature_columns)

    def _create_cyclical_features(self, profile_df: DataFrame) -> DataFrame:
        """Creates cyclical sine/cosine features from the 'registered_on' date.

        Parameters
        ----------
        profile_df : pyspark.sql.DataFrame
            The input profile DataFrame with a 'registered_on' column.

        Returns
        -------
        pyspark.sql.DataFrame
            The DataFrame enriched with cyclical date features.

        """
        df_with_date = profile_df.withColumn(
            "registration_date",
            F.to_date(F.col("registered_on").cast("string"), "yyyyMMdd"),
        )

        df_with_components = df_with_date.withColumn(
            "registration_month",
            F.month("registration_date"),
        ).withColumn(
            "registration_dayofweek",
            F.dayofweek("registration_date"),
        )

        df_with_cyclical = (
            df_with_components.withColumn(
                "month_sin",
                F.sin(2 * math.pi * F.col("registration_month") / 12),
            )
            .withColumn(
                "month_cos",
                F.cos(2 * math.pi * F.col("registration_month") / 12),
            )
            .withColumn(
                "dayofweek_sin",
                F.sin(2 * math.pi * F.col("registration_dayofweek") / 7),
            )
            .withColumn(
                "dayofweek_cos",
                F.cos(2 * math.pi * F.col("registration_dayofweek") / 7),
            )
        )

        return df_with_cyclical.drop("registration_date", "registration_month", "registration_dayofweek")
