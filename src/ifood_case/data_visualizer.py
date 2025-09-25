"""This module provides the DataVisualizer class, responsible for creating
visualizations to gain insights from the iFood case study data.
"""
from typing import NoReturn

import matplotlib.pyplot as plt
import pyspark.sql.functions as F
import seaborn as sns
from pyspark.sql import DataFrame
from pyspark.sql.functions import coalesce, col, concat_ws, explode, floor, when

# --- Brand Color Constants ---
IFOOD_RED = "#EA1D2C"
IFOOD_BLACK = "#3F3E3E"
IFOOD_PALETTE = [
    IFOOD_RED,
    "#FF7043",
    "#FFA726",
    "#D3D3D3",
    "#8B0000",
    "#696969",
]


class DataVisualizer:
    """A class for creating visualizations from offers, transactions,
    and customer profile data.
    """

    def __init__(self, offers: DataFrame, transactions: DataFrame, profile: DataFrame, df_joined: DataFrame) -> None:
        """Initializes the DataVisualizer object.

        Parameters
        ----------
        offers : pyspark.sql.DataFrame
            [cite_start]DataFrame with offer details (BOGO, informational, discount). [cite: 20, 21, 23]
        transactions : pyspark.sql.DataFrame
            [cite_start]DataFrame with the history of transactions and offer interactions. [cite: 35, 36]
        profile : pyspark.sql.DataFrame
            [cite_start]DataFrame with customer demographic data. [cite: 28, 29]
        df_joined : pyspark.sql.DataFrame
            A pre-joined DataFrame containing linked data from offers,
            transactions, and profiles.

        """
        self.offers = offers
        self.transactions = transactions
        self.profile = profile
        self.df_joined = df_joined

    def plot_barplot_channels(self) -> NoReturn:
        """Plots the count for the 'channels' column in two ways.

        This method serves as an entry point that calls two private methods
        to plot the count of channel combinations and also the count of
        individual channels.

        Returns
        -------
        None

        """
        self.__plot_barplot_channels_joined()
        self.__plot_barplot_channels_separated()

    def __plot_barplot_channels_joined(self) -> NoReturn:
        """Plots the count of channel combinations (e.g., 'email, web')."""
        channels_pd = (
            self.offers.withColumn("channels_str", concat_ws(", ", col("channels")))
            .groupBy("channels_str")
            .count()
            .orderBy("count", ascending=False)
            .toPandas()
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(data=channels_pd, x="channels_str", y="count", color=IFOOD_RED)
        plt.title("Count by Channel Combination", color=IFOOD_BLACK)
        plt.xlabel("Channel Combination", color=IFOOD_BLACK)
        plt.ylabel("Count", color=IFOOD_BLACK)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def __plot_barplot_channels_separated(self) -> NoReturn:
        """Plots the count of each individual offer channel."""
        channels_pd = (
            self.offers.withColumn("channel_exp", explode(col("channels")))
            .groupBy("channel_exp")
            .count()
            .orderBy("count", ascending=False)
            .toPandas()
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(data=channels_pd, x="channel_exp", y="count", color=IFOOD_RED)
        plt.title("Count by Individual Channel", color=IFOOD_BLACK)
        plt.xlabel("Channel", color=IFOOD_BLACK)
        plt.ylabel("Count", color=IFOOD_BLACK)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def plot_histogram_age(self) -> NoReturn:
        """Plots the age distribution from the profile data. [cite: 30]

        Returns
        -------
        None

        """
        age_pd = self.profile.select(col("age")).toPandas()

        plt.figure(figsize=(10, 6))
        sns.histplot(data=age_pd, x="age", color=IFOOD_RED)
        plt.title("Age Distribution", color=IFOOD_BLACK)
        plt.xlabel("Age", color=IFOOD_BLACK)
        plt.ylabel("Count", color=IFOOD_BLACK)
        plt.tight_layout()
        plt.show()

    def plot_histogram_credit_card_limit(self) -> NoReturn:
        """Plots the credit card limit distribution from the profile data. [cite: 34]

        Returns
        -------
        None

        """
        limit_pd = self.profile.select(col("credit_card_limit")).toPandas()

        plt.figure(figsize=(10, 6))
        sns.histplot(data=limit_pd, x="credit_card_limit", color=IFOOD_RED)
        plt.title("Credit Card Limit Distribution", color=IFOOD_BLACK)
        plt.xlabel("Limit", color=IFOOD_BLACK)
        plt.ylabel("Count", color=IFOOD_BLACK)
        plt.tight_layout()
        plt.show()

    def plot_barplot_gender(self) -> NoReturn:
        """Plots the count of each gender in the profile data. [cite: 32]

        Returns
        -------
        None

        """
        gender_pd = (
            self.profile.select(col("gender"))
            .groupBy("gender")
            .count()
            .na.fill({"gender": "Unknown"})
            .toPandas()
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(data=gender_pd, x="gender", y="count", color=IFOOD_RED)
        plt.title("Count of Profiles by Gender", color=IFOOD_BLACK)
        plt.xlabel("Gender", color=IFOOD_BLACK)
        plt.ylabel("Count", color=IFOOD_BLACK)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def plot_barplot_event(self) -> NoReturn:
        """Plots the count of each event type in the transactions data. [cite: 37]

        Returns
        -------
        None

        """
        event_pd = (
            self.transactions.select(col("event"))
            .groupBy("event")
            .count()
            .na.fill({"event": "Unknown"})
            .toPandas()
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(data=event_pd, x="event", y="count", color=IFOOD_RED)
        plt.title("Count of Event Types in Transactions", color=IFOOD_BLACK)
        plt.xlabel("Event Type", color=IFOOD_BLACK)
        plt.ylabel("Count", color=IFOOD_BLACK)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def plot_histogram_age_credit_card_limit(self) -> NoReturn:
        """Plots the credit card limit distribution segmented by age group.

        Returns
        -------
        None

        """
        limit_age_pd = self.profile.select(
            col("age_group"), col("credit_card_limit"),
        ).toPandas()

        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=limit_age_pd,
            x="credit_card_limit",
            hue="age_group",
            palette=IFOOD_PALETTE,
        )
        plt.title("Credit Card Limit Distribution by Age Group", color=IFOOD_BLACK)
        plt.xlabel("Limit", color=IFOOD_BLACK)
        plt.ylabel("Count", color=IFOOD_BLACK)
        plt.tight_layout()
        plt.show()

    def plot_histogram_gender_credit_card_limit(self) -> NoReturn:
        """Plots the credit card limit distribution segmented by gender.

        Returns
        -------
        None

        """
        limit_gender_pd = (
            self.profile.select(col("gender"), col("credit_card_limit"))
            .na.fill({"gender": "Unknown"})
            .toPandas()
        )

        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=limit_gender_pd,
            x="credit_card_limit",
            hue="gender",
            palette=IFOOD_PALETTE,
        )
        plt.title("Credit Card Limit Distribution by Gender", color=IFOOD_BLACK)
        plt.xlabel("Limit", color=IFOOD_BLACK)
        plt.ylabel("Count", color=IFOOD_BLACK)
        plt.tight_layout()
        plt.show()

    def plot_offer_effectiveness_by_profile(self) -> NoReturn:
        """Plots the count of completed offers by offer type, segmented by
        gender and age group.

        This analysis helps to identify which customer profiles are more
        receptive to certain types of offers.

        Returns
        -------
        None

        """
        completed_offers_pd = (
            self.df_joined.filter(col("event") == "offer completed")
            .withColumn("offer_type_union", coalesce("offer_type_1", "offer_type_2"))
            .na.fill({"gender": "Unknown"})
            .groupBy("offer_type_union", "age_group", "gender")
            .count()
            .orderBy("age_group", "gender")
            .toPandas()
        )

        g = sns.catplot(
            data=completed_offers_pd,
            x="age_group",
            y="count",
            hue="offer_type_union",
            col="gender",
            kind="bar",
            palette=IFOOD_PALETTE,
            height=6,
            aspect=1.2,
        )

        g.fig.suptitle(
            "Offer Effectiveness by Gender and Age Group",
            y=1.03,
            fontsize=16,
            color=IFOOD_BLACK,
        )
        g.set_axis_labels("Age Group", "# of Completed Offers")
        g.set_titles("Gender: {col_name}")
        g.despine(left=True)
        plt.tight_layout()
        plt.show()

    def plot_conversion_rate_by_channel(self) -> NoReturn:
        """Calculates and plots the conversion rate (completed/viewed) for each
        marketing channel.

        This analysis is key to optimizing marketing investment by focusing
        on channels that generate the highest return.

        Returns
        -------
        None

        """
        viewed_counts = (
            self.transactions.filter(col("event") == "offer viewed")
            .groupBy("offer_received_viewed")
            .count()
            .withColumnRenamed("count", "viewed_count")
        )

        completed_counts = (
            self.transactions.filter(col("event") == "offer completed")
            .groupBy("offer_completed")
            .count()
            .withColumnRenamed("count", "completed_count")
        )

        conversion_df = (
            self.offers.join(viewed_counts, self.offers.id == viewed_counts.offer_received_viewed, "left")
            .join(completed_counts, self.offers.id == completed_counts.offer_completed, "left")
            .na.fill(0)
        )

        conversion_by_channel_pd = (
            conversion_df.withColumn("channel", explode(col("channels")))
            .groupBy("channel")
            .agg(
                F.sum("viewed_count").alias("total_viewed"),
                F.sum("completed_count").alias("total_completed"),
            )
            .withColumn(
                "conversion_rate",
                when(col("total_viewed") > 0, (col("total_completed") / col("total_viewed")) * 100).otherwise(0),
            )
            .orderBy("conversion_rate", ascending=False)
            .toPandas()
        )

        plt.figure(figsize=(12, 7))
        ax = sns.barplot(
            data=conversion_by_channel_pd,
            x="channel",
            y="conversion_rate",
            color=IFOOD_RED,
        )

        plt.title("Conversion Rate by Marketing Channel (%)", fontsize=16, color=IFOOD_BLACK)
        plt.xlabel("Channel", fontsize=12, color=IFOOD_BLACK)
        plt.ylabel("Conversion Rate (%)", fontsize=12, color=IFOOD_BLACK)

        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.1f}%",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 9),
                textcoords="offset points",
                color=IFOOD_BLACK,
            )

        if not conversion_by_channel_pd.empty:
            plt.ylim(0, max(conversion_by_channel_pd["conversion_rate"]) * 1.15)
        plt.tight_layout()
        plt.show()

    def plot_transaction_value_by_offer_usage(self) -> NoReturn:
        """Compares the transaction value distribution (ticket size) between
        customers who have completed offers and those who have not.

        This analysis assesses the impact of offers on customer spending behavior.

        Returns
        -------
        None

        """
        users_who_completed = (
            self.transactions.filter(col("event") == "offer completed")
            .select("account_id")
            .distinct()
            .withColumn("completed_offer", F.lit(True))
        )

        transactions_only = self.transactions.filter(col("event") == "transaction")

        transactions_with_user_type_pd = (
            transactions_only.join(users_who_completed, "account_id", "left")
            .na.fill({"completed_offer": False})
            .withColumn("user_type", when(col("completed_offer") == True, "Uses Offers").otherwise("Does Not Use Offers"))
            .toPandas()
        )

        plt.figure(figsize=(10, 8))
        sns.histplot(
            data=transactions_with_user_type_pd,
            x="amount",
            hue="user_type",
            palette=[IFOOD_RED, IFOOD_BLACK],
            log_scale=True,
            element="step",
            kde=True,
        )
        plt.title("Transaction Value Distribution (Log Scale)", fontsize=16, color=IFOOD_BLACK)
        plt.xlabel("Transaction Value ($) - Log Scale", fontsize=12, color=IFOOD_BLACK)
        plt.ylabel("Count", fontsize=12, color=IFOOD_BLACK)
        plt.tight_layout()
        plt.show()

    def plot_informational_offer_impact(self) -> NoReturn:
        """Analyzes and plots the impact of informational offers on transaction
        value, removing outliers for better visualization clarity.

        Returns
        -------
        None

        """
        informational_offer_ids = [
            row.id
            for row in self.offers.filter(col("offer_type") == "informational")
            .select("id")
            .collect()
        ]
        info_views_df = (
            self.transactions.filter(
                (col("event") == "offer viewed")
                & (col("offer_received_viewed").isin(informational_offer_ids)),
            )
            .alias("v")
            .join(self.offers.alias("o"), col("v.offer_received_viewed") == col("o.id"))
            .select(
                col("v.account_id"),
                col("v.time_since_test_start").alias("view_time"),
                col("o.duration"),
            )
        )
        transactions_df = self.transactions.filter(
            col("event") == "transaction",
        ).alias("t")
        influenced_transactions = info_views_df.join(
            transactions_df,
            info_views_df.account_id == transactions_df.account_id,
            "inner",
        ).filter(
            (col("t.time_since_test_start") >= col("view_time"))
            & (col("t.time_since_test_start") <= col("view_time") + col("duration")),
        )
        final_df = influenced_transactions.join(
            self.profile, on=col("t.account_id") == self.profile.id, how="left",
        ).select(
            col("t.account_id").alias("account_id"),
            "gender",
            "age_group",
            "amount",
        ).na.fill({"gender": "Unknown"})

        plot_data_pd = final_df.toPandas()

        if plot_data_pd.empty:
            print("No transactions influenced by informational offers were found.")
            return

        Q1 = plot_data_pd.groupby(["age_group", "gender"])["amount"].transform(
            lambda x: x.quantile(0.25),
        )
        Q3 = plot_data_pd.groupby(["age_group", "gender"])["amount"].transform(
            lambda x: x.quantile(0.75),
        )
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_plot_no_outliers = plot_data_pd[
            (plot_data_pd["amount"] >= lower_bound)
            & (plot_data_pd["amount"] <= upper_bound)
        ]

        g = sns.catplot(
            data=df_plot_no_outliers,
            x="age_group",
            y="amount",
            hue="gender",
            kind="boxen",
            palette=IFOOD_PALETTE,
            height=7,
            aspect=1.5,
            order=["18-25", "26-35", "36-50", "51+"],
        )

        g.fig.suptitle(
            "Transaction Value After Informational Offer (Outliers Removed)",
            y=1.03,
            fontsize=18,
        )
        g.set_axis_labels("Age Group", "Transaction Value ($)")
        g._legend.set_title("Gender")
        g.despine(left=True)
        plt.tight_layout()
        plt.show()

    def plot_engagement_over_time_by_profile(self) -> NoReturn:
        """Analyzes and plots the evolution of offer engagement (viewed and
        completed) over time (in weeks), segmented by customer profile.

        This temporal analysis helps identify trends, activity peaks, and
        potential fatigue among different customer segments.

        Returns
        -------
        None

        """
        print("Starting temporal engagement analysis by profile...")

        offer_events = self.df_joined.filter(
            col("event").isin("offer viewed", "offer completed"),
        )

        events_by_week = offer_events.withColumn(
            "week", floor(col("time_since_test_start") / 7),
        )

        engagement_by_gender_pd = (
            events_by_week.groupBy("week", "gender", "event")
            .count()
            .orderBy("week", "gender")
            .toPandas()
        )
        engagement_by_age_pd = (
            events_by_week.groupBy("week", "age_group", "event")
            .count()
            .orderBy("week", "age_group")
            .toPandas()
        )

        plt.figure(figsize=(15, 8))
        sns.lineplot(
            data=engagement_by_gender_pd,
            x="week",
            y="count",
            hue="gender",
            style="event",
            palette=IFOOD_PALETTE,
            linewidth=2.5,
        )
        plt.title("Offer Engagement Evolution by Gender", fontsize=18, color=IFOOD_BLACK)
        plt.xlabel("Week of Test", fontsize=14, color=IFOOD_BLACK)
        plt.ylabel("Number of Events", fontsize=14, color=IFOOD_BLACK)
        plt.legend(title="Legend")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(15, 8))
        sns.lineplot(
            data=engagement_by_age_pd,
            x="week",
            y="count",
            hue="age_group",
            style="event",
            palette=IFOOD_PALETTE,
            linewidth=2.5,
            hue_order=["18-25", "26-35", "36-50", "51+"],
        )
        plt.title("Offer Engagement Evolution by Age Group", fontsize=18, color=IFOOD_BLACK)
        plt.xlabel("Week of Test", fontsize=14, color=IFOOD_BLACK)
        plt.ylabel("Number of Events", fontsize=14, color=IFOOD_BLACK)
        plt.legend(title="Legend")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
