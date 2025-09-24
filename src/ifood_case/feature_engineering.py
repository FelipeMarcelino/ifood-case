import math

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class FeatureEngineering:
    def __init__(self, offers: DataFrame, transactions: DataFrame, profile: DataFrame):
        self.offers = offers
        self.transactions = transactions
        self.profile = profile

    def transform(self):
        point_in_time_features = self.__point_in_time_features()
        last_offer_features = self.__create_last_offer_viewed_feature()
        profile_features = self.__create_profile_features()
        offers_features = self.__create_offers_features()
        target = self.__create_target_variable()

        # Primeiro join (entre target e features)
        df_final = (
            target.alias("t")
            .join(
                point_in_time_features.alias("p"),
                on=(
                    (F.col("t.offer_id") == F.col("p.offer_id_received"))
                    & (F.col("t.account_id") == F.col("p.account_id"))
                    & (F.col("t.time_received") == F.col("p.time_received"))
                ),
                how="left",
            )
            .drop(F.col("p.account_id"), F.col("p.offer_id_received"), F.col("p.time_received"))
        )

        # Segundo join (adicionando a última oferta vista)
        df_final = (
            df_final.alias("f")
            .join(
                last_offer_features.alias("l"),
                on=(
                    (F.col("f.offer_id") == F.col("l.offer_id_received"))
                    & (F.col("f.account_id") == F.col("l.account_id"))
                    & (F.col("f.time_received") == F.col("l.time_received"))
                ),
                how="left",
            )
            .drop(F.col("l.time_received"), F.col("l.account_id"), F.col("l.offer_id_received"))
        )

        # Terceiro join entre profile features
        df_final = (
            df_final.alias("f")
            .join(
                profile_features.alias("p"),
                on=(F.col("f.account_id") == F.col("p.id")),
                how="left",
            )
            .drop(F.col("p.id"))
        )

        # Quarto join, features de oferta
        df_final = df_final.alias("f").join(offers_features.alias("o"), on=(F.col("f.offer_id") == F.col("o.id")),
                                 how="left").drop(F.col("o.id"))

        return df_final

    def __clean_age(self):
        # Substitui o valor de 118 por nulo para o dataset profile
        self.profile = self.profile.withColumn("age", F.when(F.col("age") == 118, None).otherwise(F.col("age")))

    def __create_target_variable(self):
        """Função responsável por criar o target do dataset relacionado as ofertas de disconto,
        bogo e informacional.
        """
        # Dataframe contendo as promoções que foram completadas relacionada a bogo e disconto
        bogo_discount_success = (
            self.transactions.filter(F.col("event") == "offer completed")
            .select("account_id", F.col("offer_completed").alias("successful_offer_id"))
            .distinct()
        )

        # Capturando as ofertas informacionais dentro do dataset offer
        informational_offer_ids = [
            row.id for row in self.offers.filter(F.col("offer_type") == "informational").select("id").collect()
        ]

        # Pegando todas as linhas contendo ofertas informacionais
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

        # Pegando transações que foram feitas contendo um valor
        all_transactions = self.transactions.filter(F.col("event") == "transaction").alias("t")

        # Cria o dataframe de informacional sucesso, transações que foram feitas dentro do intervalo
        # de duração da oferta informacional é considerado uma oferta de sucesso
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

        # Junto com as informações de bogo/disconto com informacionais
        all_successful_offers = bogo_discount_success.unionByName(informational_success)

        # Cria o dataset de account_id/offer_id básico
        offers_received_base = (
            self.transactions.filter(F.col("event") == "offer received")
            .select(
                "account_id",
                F.col("offer_received_viewed").alias("offer_id"),
                F.col("time_since_test_start").alias("time_received"),
            )
            .distinct()
        )

        # O join para criar o target
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
            .select(
                F.col("base.account_id"),
                F.col("base.offer_id"),
                F.col("base.time_received"),
                "target",
            )
        )

        return target_dataset

    def __point_in_time_features(self) -> DataFrame:
        """Cria um dataset com features calculadas "point-in-time" para cada oferta recebida."""
        window_spec = Window.partitionBy("account_id").orderBy("time_since_test_start")

        features_over_time = (
            self.transactions.withColumn(
                "amount_if_transaction",
                F.when(F.col("event") == "transaction", F.col("amount")).otherwise(None),
            )
            .withColumn(
                "spend_cumulative",
                F.sum("amount_if_transaction").over(window_spec),
            )
            .withColumn(
                "transactions_cumulative_count",
                F.sum(F.when(F.col("event") == "transaction", 1).otherwise(0)).over(window_spec),
            )
            .withColumn(
                "avg_ticket_cumulative",
                F.avg("amount_if_transaction").over(window_spec),
            )
            .withColumn(
                "max_ticket_cumulative",
                F.max("amount_if_transaction").over(window_spec),
            )
            .withColumn(
                "min_ticket_cumulative",
                F.min("amount_if_transaction").over(window_spec),
            )
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
            features_over_time.withColumn(
                "total_spend_before",
                F.lag("spend_cumulative", 1, 0).over(window_spec),
            )
            .withColumn(
                "transaction_count_before",
                F.lag("transactions_cumulative_count", 1, 0).over(window_spec),
            )
            .withColumn(
                "avg_ticket_before",
                F.lag("avg_ticket_cumulative", 1, 0).over(window_spec),
            )
            .withColumn(
                "max_ticket_before",
                F.lag("max_ticket_cumulative", 1, 0).over(window_spec),
            )
            .withColumn(
                "min_ticket_before",
                F.lag("min_ticket_cumulative", 1, 0).over(window_spec),
            )
            .withColumn(
                "offers_viewed_count_before",
                F.lag("offers_viewed_cumulative_count", 1, 0).over(window_spec),
            )
            .withColumn(
                "offers_completed_count_before",
                F.lag("offers_completed_cumulative_count", 1, 0).over(window_spec),
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

    def __create_last_offer_viewed_feature(self) -> DataFrame:
        """Para cada oportunidade de oferta, encontra qual foi a última oferta visualizada
        pelo cliente antes daquela oportunidade.

        Parameters
        ----------
        transactions_parsed : pyspark.sql.DataFrame
            O log de todas as transações e eventos de oferta.
        opportunities : pyspark.sql.DataFrame
            DataFrame base com (account_id, offer_id_received, time_received).

        Returns
        -------
        pyspark.sql.DataFrame
            O DataFrame de oportunidades enriquecido com a coluna 'last_offer_viewed_id'.

        """
        opportunities = self.transactions.filter(F.col("event") == "offer received").select(
            "account_id",
            F.col("offer_received_viewed").alias("offer_id_received"),
            F.col("time_since_test_start").alias("time_received"),
        )

        # 1. Isolar todos os eventos de "oferta visualizada"
        viewed_events = self.transactions.filter(F.col("event") == "offer viewed").select(
            F.col("account_id"),
            F.col("offer_received_viewed").alias("offer_id_viewed"),
            F.col("time_since_test_start").alias("time_viewed"),
        )

        # 2. Join temporal: juntar cada oportunidade com TODAS as visualizações anteriores do mesmo cliente
        joined_df = opportunities.alias("opp").join(
            viewed_events.alias("v"),
            (F.col("opp.account_id") == F.col("v.account_id"))
            & (F.col("opp.time_received") > F.col("v.time_viewed")),  # A visualização deve ser ANTERIOR
            "left",
        )

        # 3. Usar uma Window Function para encontrar a MAIS RECENTE das visualizações anteriores
        window_spec = Window.partitionBy("opp.account_id", "opp.offer_id_received", "opp.time_received").orderBy(
            F.col("v.time_viewed").desc(),
        )

        ranked_views = joined_df.withColumn("rank", F.row_number().over(window_spec))

        # 4. A última visualização é aquela com rank = 1
        last_viewed_offer = ranked_views.filter(F.col("rank") == 1).select(
            "opp.account_id",
            "opp.offer_id_received",
            "opp.time_received",
            F.col("v.offer_id_viewed").alias("last_offer_viewed_id"),
        )

        last_viewed_offer_type_df = last_viewed_offer.join(
            self.offers.alias("o_details"),
            last_viewed_offer.last_offer_viewed_id == F.col("o_details.id"),
            "left",  # Left join para não perder clientes que nunca viram uma oferta
        ).select(
            "account_id",
            "offer_id_received",
            "time_received",
            F.col("o_details.offer_type").alias("last_offer_viewed_type"),
        )

        return last_viewed_offer_type_df

    def __create_cyclical_features(self):
        """Cria features cíclicas de seno e cosseno a partir da data de cadastro.

        Returns
        -------
        pyspark.sql.DataFrame
            O DataFrame enriquecido com as features cíclicas.

        """
        self.profile = self.profile.withColumn(
            "registration_date",
            F.to_date(F.col("registered_on").cast("string"), "yyyyMMdd"),
        )

        self.profile = self.profile.withColumn(
            "registration_month", F.month("registration_date"),
        ).withColumn(
            "registration_dayofweek", F.dayofweek("registration_date"), # Domingo=1, Sábado=7
        )

        self.profile = self.profile.withColumn(
            "month_sin",
            F.sin(2 * math.pi * F.col("registration_month") / 12),
        ).withColumn(
            "month_cos",
            F.cos(2 * math.pi * F.col("registration_month") / 12),
        ).withColumn(
            "dayofweek_sin",
            F.sin(2 * math.pi * F.col("registration_dayofweek") / 7),
        ).withColumn(
            "dayofweek_cos",
            F.cos(2 * math.pi * F.col("registration_dayofweek") / 7),
        )

        self.profile.drop("registration_date", "registration_month", "registration_dayofweek")

    def __create_profile_features(self) -> DataFrame:
        self.__clean_age()
        self.__create_cyclical_features()

        selected_profile_features = self.profile.select(
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

    def __explode_channels(self):
        channels_df = self.offers.select(F.explode("channels").alias("channel"))
        distinct_channels = [row.channel for row in channels_df.distinct().collect()]

        for channel in distinct_channels:
            self.offers = self.offers.withColumn(
                f"{channel}",
                F.when(F.array_contains(F.col("channels"), channel), 1).otherwise(0),
            )

    def __ratio_discount(self):
        self.offers = self.offers.withColumn(
            "reward_ratio",
            F.when(
                F.col("min_value") > 0,
                F.col("discount_value") / F.col("min_value"),
            ).otherwise(0),
        )

    def __create_offers_features(self) -> DataFrame:
        self.__explode_channels()
        self.__ratio_discount()

        selected_offer_features = self.offers.select(
            F.col("discount_value"),
            F.col("duration"),
            F.col("min_value"),
            F.col("offer_type"),
            F.col("email"),
            F.col("social"),
            F.col("mobile"),
            F.col("web"),
            F.col("id"),
        )

        return selected_offer_features
