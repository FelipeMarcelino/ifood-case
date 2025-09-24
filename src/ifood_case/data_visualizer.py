"""Módulo responsável por criações das visualizações para insigts sobre os dados."""

import matplotlib.pyplot as plt
import pyspark.sql.functions as F
import seaborn as sns
from pyspark.sql import DataFrame
from pyspark.sql.functions import coalesce, col, concat_ws, explode, floor, when

IFOOD_RED = "#EA1D2C"
IFOOD_BLACK = "#3F3E3E"
IFOOD_GRAY_LIGHT = "#E0E0E0"
IFOOD_ORANGE = "#FF7C00"
IFOOD_LIGHT_RED = "#F9868D"


IFOOD_PALETTE = [
    IFOOD_RED,
    "#FF7043",
    "#FFA726",
    "#D3D3D3",
    "#8B0000",
    "#696969",
]


class DataVisualizer:
    """Uma classe para criar visualizações a partir de dados de ofertas,
    transações e perfis de clientes da Starbucks.
    """

    def __init__(self, offers: DataFrame, transactions: DataFrame, profile: DataFrame, df_joined: DataFrame) -> None:
        """Inicializa o objeto DataVisualizer.

        Parameters
        ----------
        offers : pyspark.sql.DataFrame
            DataFrame com detalhes das ofertas (BOGO, informational, discount).
        transactions : pyspark.sql.DataFrame
            DataFrame com o histórico de transações e interações com ofertas.
        profile : pyspark.sql.DataFrame
            DataFrame com os dados demográficos dos clientes.
        df_joined: pyspark.sql.DataFrame
            DataFrame com os dados interligado entre si, contendo dados de offers, transactions e profile.

        Attributes
        ----------
        offers : pyspark.sql.DataFrame
            Armazena o DataFrame de ofertas fornecido.
        transactions : pyspark.sql.DataFrame
            Armazena o DataFrame de transações fornecido.
        profile : pyspark.sql.DataFrame
            Armazena o DataFrame de perfis fornecido.
        df_joined: pyspark.sql.DataFrame
            Armazena o DataFrame de dados interligados

        """
        self.offers = offers
        self.transactions = transactions
        self.profile = profile
        self.df_joined = df_joined

    def plot_barplot_channels(self) -> None:
        """Plota a contagem para a coluna 'channels' de duas formas.

        Este método serve como um ponto de entrada que chama dois métodos
        privados para plotar a contagem de combinações de canais e também
        a contagem de canais individuais.

        Returns
        -------
        None

        """
        self.__plot_barplot_channels_joined()
        self.__plot_barplot_channels_separated()

    def __plot_barplot_channels_joined(self) -> None:
        """Plota a contagem de combinações de canais (ex: 'email, web')."""
        channels_pd = (
            self.offers.withColumn("channels_str", concat_ws(", ", col("channels")))
            .groupBy("channels_str")
            .count()
            .orderBy("count", ascending=False)
            .toPandas()
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(data=channels_pd, x="channels_str", y="count", color=IFOOD_RED)
        plt.title("Contagem por Combinação de Canais", color=IFOOD_BLACK)
        plt.xlabel("Combinação de Canais", color=IFOOD_BLACK)
        plt.ylabel("Contagem", color=IFOOD_BLACK)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def __plot_barplot_channels_separated(self) -> None:
        """Plota a contagem de cada canal de oferta individualmente."""
        channels_exploded = self.offers.withColumn("channels_exp", explode(col("channels")))
        channels_counts = channels_exploded.groupBy("channels_exp").count().orderBy("count", ascending=False)
        channels_pd = channels_counts.toPandas()

        plt.figure(figsize=(10, 6))
        sns.barplot(data=channels_pd, x="channels_exp", y="count", color=IFOOD_RED)
        plt.title("Contagem por Canal Individual", color=IFOOD_BLACK)
        plt.xlabel("Canal", color=IFOOD_BLACK)
        plt.ylabel("Contagem", color=IFOOD_BLACK)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def plot_histogram_age(self) -> None:
        """Plota a distribuição de idade por perfil."""
        age = self.profile.select(col("age")).toPandas()

        plt.figure(figsize=(10, 6))
        sns.histplot(data=age, x="age", color=IFOOD_RED)
        plt.title("Distribuição de idade", color=IFOOD_BLACK)
        plt.xlabel("Idade", color=IFOOD_BLACK)
        plt.ylabel("Contagem", color=IFOOD_BLACK)
        plt.tight_layout()
        plt.show()

    def plot_histogram_credit_card_limit(self) -> None:
        """Plota a distribuição de limite de cartão de crédito por perfil."""
        limit = self.profile.select(col("credit_card_limit")).toPandas()

        plt.figure(figsize=(10, 6))
        sns.histplot(data=limit, x="credit_card_limit", color=IFOOD_RED)
        plt.title("Distribuição de limite de cartão de crédito", color=IFOOD_BLACK)
        plt.xlabel("Limite", color=IFOOD_BLACK)
        plt.ylabel("Contagem", color=IFOOD_BLACK)
        plt.tight_layout()
        plt.show()

    def plot_barplot_gender(self) -> None:
        """Plota a contagem de gênero dentro dos perfis."""
        gender = self.profile.select(col("gender")).groupBy("gender").count().na.fill({"gender": "Nulo"}).toPandas()

        plt.figure(figsize=(10, 6))
        sns.barplot(data=gender, x="gender", y="count", color=IFOOD_RED)
        plt.title("Contagem de perfis de cada gênero", color=IFOOD_BLACK)
        plt.xlabel("Gênero", color=IFOOD_BLACK)
        plt.ylabel("Contagem", color=IFOOD_BLACK)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def plot_barplot_event(self) -> None:
        """Plota a contagem de tipos de events dentro de transações."""
        event = self.transactions.select(col("event")).groupBy("event").count().na.fill({"event": "Nulo"}).toPandas()

        plt.figure(figsize=(10, 6))
        sns.barplot(data=event, x="event", y="count", color=IFOOD_RED)
        plt.title("Contagem de tipos de eventos nas transações", color=IFOOD_BLACK)
        plt.xlabel("Eventos", color=IFOOD_BLACK)
        plt.ylabel("Contagem", color=IFOOD_BLACK)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def plot_histogram_age_credit_card_limit(self) -> None:
        limit_age = self.profile.select(col("age_group"), col("credit_card_limit")).toPandas()

        plt.figure(figsize=(10, 6))
        sns.histplot(data=limit_age, x="credit_card_limit", hue="age_group", palette=IFOOD_PALETTE)
        plt.title("Distribuição de limite de cartão de crédito por idade", color=IFOOD_BLACK)
        plt.xlabel("Limite", color=IFOOD_BLACK)
        plt.ylabel("Contagem", color=IFOOD_BLACK)
        plt.tight_layout()
        plt.show()

    def plot_histogram_gender_credit_card_limit(self) -> None:
        limit_gender = (
            self.profile.select(col("gender"), col("credit_card_limit")).na.fill({"gender": "Nulo"}).toPandas()
        )

        plt.figure(figsize=(10, 6))
        sns.histplot(data=limit_gender, x="credit_card_limit", hue="gender", palette=IFOOD_PALETTE)
        plt.title("Distribuição de limite de cartão de crédito por gênero", color=IFOOD_BLACK)
        plt.xlabel("Limite", color=IFOOD_BLACK)
        plt.ylabel("Contagem", color=IFOOD_BLACK)
        plt.tight_layout()
        plt.show()

    def plot_offer_effectiveness_by_profile(self) -> None:
        """Plota a contagem de ofertas completadas por tipo de oferta,
        segmentado por gênero e faixa etária.

        Esta análise ajuda a identificar quais perfis de clientes são mais
        receptivos a determinados tipos de oferta.
        """
        completed_offers_pd = (
            self.df_joined.filter(col("event") == "offer completed")
            .withColumn("offer_type_union", coalesce("offer_type_1", "offer_type_2"))
            .na.fill({"gender": "Nulo"})
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

        g.fig.suptitle("Efetividade da Oferta por Gênero e Faixa Etária", y=1.03, fontsize=16, color=IFOOD_BLACK)
        g.set_axis_labels("Faixa Etária", "Nº de Ofertas Completadas")
        g.set_titles("Gênero: {col_name}")
        g.despine(left=True)
        plt.tight_layout()
        plt.show()

    def plot_conversion_rate_by_channel(self) -> None:
        """Calcula e plota a taxa de conversão (completadas/visualizadas) para
        cada canal de marketing.

        Esta análise é fundamental para otimizar o investimento em marketing,
        focando nos canais que geram maior retorno.
        """
        # Contagem de ofertas visualizadas por ID
        viewed_counts = (
            self.transactions.filter(col("event") == "offer viewed")
            .groupBy("offer_received_viewed")
            .count()
            .withColumnRenamed("count", "viewed_count")
        )

        # Contagem de ofertas completadas por ID
        completed_counts = (
            self.transactions.filter(col("event") == "offer completed")
            .groupBy("offer_completed")
            .count()
            .withColumnRenamed("count", "completed_count")
        )

        # Juntamos as contagens com os detalhes das ofertas
        conversion_df = (
            self.offers.join(viewed_counts, self.offers.id == viewed_counts.offer_received_viewed, "left")
            .join(completed_counts, self.offers.id == completed_counts.offer_completed, "left")
            .na.fill(0)
        )

        # Explodimos os canais e calculamos a conversão
        conversion_by_channel_pd = (
            conversion_df.withColumn("channel", explode(col("channels")))
            .groupBy("channel")
            .agg(
                F.sum("viewed_count").alias("total_viewed"),
                F.sum("completed_count").alias("total_completed"),
            )
            .withColumn("conversion_rate", (col("total_completed") / col("total_viewed")) * 100)
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

        plt.title("Taxa de Conversão por Canal de Marketing (%)", fontsize=16, color=IFOOD_BLACK)
        plt.xlabel("Canal", fontsize=12, color=IFOOD_BLACK)
        plt.ylabel("Taxa de Conversão (%)", fontsize=12, color=IFOOD_BLACK)

        # Adicionar rótulos de dados nas barras
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

        plt.ylim(0, max(conversion_by_channel_pd["conversion_rate"]) * 1.15)
        plt.tight_layout()
        plt.show()

    def plot_transaction_value_by_offer_usage(self) -> None:
        """Compara a distribuição do valor das transações (ticket médio) entre
        clientes que completaram ofertas e os que não completaram.

        Essa análise avalia o impacto das ofertas no comportamento de gastos do cliente.
        """
        # Identificar clientes que completaram pelo menos uma oferta
        users_who_completed = (
            self.transactions.filter(col("event") == "offer completed")
            .select("account_id")
            .distinct()
            .withColumn("completed_offer", F.lit(True))
        )

        # Pegar apenas as transações (que têm valor de 'amount')
        transactions_only = self.transactions.filter(col("event") == "transaction")

        # Juntar para saber se a transação foi de um usuário que completa ofertas
        transactions_with_user_type = (
            transactions_only.join(
                users_who_completed,
                "account_id",
                "left",
            )
            .na.fill({"completed_offer": False})
            .withColumn("user_type", when(col("completed_offer") == True, "Usa Ofertas").otherwise("Não Usa Ofertas"))
            .toPandas()
        )

        plt.figure(figsize=(10, 8))
        sns.histplot(
            data=transactions_with_user_type,
            x="amount",
            hue="user_type",  # Cria um histograma para cada tipo de usuário
            palette=[IFOOD_RED, IFOOD_BLACK],
            log_scale=True,  # A MÁGICA ACONTECE AQUI!
            element="step",  # Estilo de plotagem para sobreposição
            kde=True,
        )  # Adiciona uma linha de densidade para suavizar
        plt.title("Distribuição do Valor da Transação (Escala Log)", fontsize=16, color=IFOOD_BLACK)
        plt.xlabel("Valor da Transação (R$) - Escala Logarítmica", fontsize=12, color=IFOOD_BLACK)
        plt.ylabel("Contagem", fontsize=12, color=IFOOD_BLACK)
        plt.tight_layout()
        plt.show()


    def plot_informational_offer_impact(self) -> None:
        """Analisa e plota o impacto das ofertas informacionais no valor das transações,
        removendo outliers para melhor clareza da visualização.
        """
        # 1. A lógica de preparação de dados em PySpark continua a mesma
        informational_offer_ids = [
            row.id for row in self.offers.filter(col("offer_type") == "informational").select("id").collect()
        ]
        info_views_df = (
            self.transactions.filter(
                (col("event") == "offer viewed") &
                (col("offer_received_viewed").isin(informational_offer_ids)),
            )
            .alias("v")
            .join(self.offers.alias("o"), col("v.offer_received_viewed") == col("o.id"))
            .select(
                col("v.account_id"),
                col("v.time_since_test_start").alias("view_time"),
                col("o.duration"),
            )
        )
        transactions_df = self.transactions.filter(col("event") == "transaction").alias("t")
        influenced_transactions = info_views_df.join(
            transactions_df,
            info_views_df.account_id == transactions_df.account_id,
            "inner",
        ).filter(
            (col("t.time_since_test_start") >= col("view_time")) &
            (col("t.time_since_test_start") <= col("view_time") + col("duration")),
        )
        final_df = (
            influenced_transactions
            .join(self.profile, on=col("t.account_id") == self.profile.id, how="left")
            .select(
                col("t.account_id").alias("account_id"),
                "gender",
                "age_group",
                "amount",
            ).na.fill({"gender":"Nulo"})
        )

        # 2. Coletar o resultado para o Pandas
        plot_data_pd = final_df.toPandas()



        # --- NOVO: Lógica para Remover Outliers por Grupo (age_group, gender) ---
        # Calcula Q1 e Q3 para cada grupo
        Q1 = plot_data_pd.groupby(["age_group", "gender"])["amount"].transform(lambda x: x.quantile(0.25))
        Q3 = plot_data_pd.groupby(["age_group", "gender"])["amount"].transform(lambda x: x.quantile(0.75))
        IQR = Q3 - Q1

        # Define os limites inferior e superior
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filtra o DataFrame original
        df_plot_no_outliers = plot_data_pd[(plot_data_pd["amount"] >= lower_bound) & (plot_data_pd["amount"] <= upper_bound)]
        # --- FIM DA SEÇÃO DE REMOÇÃO DE OUTLIERS ---

        # 3. Plotar o gráfico com o DataFrame FILTRADO
        g = sns.catplot(
            data=df_plot_no_outliers, # Usando o DataFrame filtrado
            x="age_group",
            y="amount",
            hue="gender",
            kind="boxen",
            palette=IFOOD_PALETTE,
            height=7,
            aspect=1.5,
            order=["18-25", "26-35", "36-50", "51+" ],
        )

        g.fig.suptitle("Valor Gasto em Transações Pós-Oferta Informativa (sem Outliers)", y=1.03, fontsize=18)
        g.set_axis_labels("Faixa Etária", "Valor da Transação (R$)")
        g._legend.set_title("Gênero")
        g.despine(left=True)
        plt.tight_layout()
        plt.show()

    def plot_engagement_over_time_by_profile(self) -> None:
        """Analisa e plota a evolução do engajamento com ofertas (visualizadas e
        completadas) ao longo do tempo (em semanas), segmentado por perfil de
        cliente (gênero e faixa etária).

        Esta análise temporal ajuda a identificar tendências, picos de atividade
        e possível fadiga de diferentes segmentos de clientes.
        """
        print("Iniciando a análise de engajamento temporal por perfil...")

        # 1. Usar o DataFrame mestre com os dados de perfil já unidos
        #    e filtrar apenas eventos de oferta.
        offer_events = self.df_joined.filter(
            col("event").isin("offer viewed", "offer completed"),
        )

        # 2. Binarizar o tempo em semanas para suavizar o gráfico.
        #    (time_since_test_start está em dias, então dividimos por 7)
        events_by_week = offer_events.withColumn(
            "week", floor(col("time_since_test_start") / 7),
        )

        # 3. Preparar dados para o plot por GÊNERO
        engagement_by_gender_pd = (
            events_by_week.groupBy("week", "gender", "event")
            .count()
            .orderBy("week", "gender")
            .toPandas()
        )

        # 4. Preparar dados para o plot por FAIXA ETÁRIA
        engagement_by_age_pd = (
            events_by_week.groupBy("week", "age_group", "event")
            .count()
            .orderBy("week", "age_group")
            .toPandas()
        )

        # --- Plot 1: Engajamento ao Longo do Tempo por Gênero ---
        plt.figure(figsize=(15, 8))
        sns.lineplot(
            data=engagement_by_gender_pd,
            x="week",
            y="count",
            hue="gender",  # Uma cor de linha para cada gênero
            style="event", # Um estilo de linha para 'viewed' vs 'completed'
            palette=IFOOD_PALETTE,
            linewidth=2.5,
        )
        plt.title("Evolução do Engajamento com Ofertas por Gênero", fontsize=18, color=IFOOD_BLACK)
        plt.xlabel("Semana do Teste", fontsize=14, color=IFOOD_BLACK)
        plt.ylabel("Número de Eventos", fontsize=14, color=IFOOD_BLACK)
        plt.legend(title="Legenda")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

        # --- Plot 2: Engajamento ao Longo do Tempo por Faixa Etária ---
        plt.figure(figsize=(15, 8))
        sns.lineplot(
            data=engagement_by_age_pd,
            x="week",
            y="count",
            hue="age_group", # Uma cor de linha para cada faixa etária
            style="event",
            palette=IFOOD_PALETTE,
            linewidth=2.5,
            hue_order=["18-25", "26-35", "36-50", "51+" ],
        )
        plt.title("Evolução do Engajamento com Ofertas por Faixa Etária", fontsize=18, color=IFOOD_BLACK)
        plt.xlabel("Semana do Teste", fontsize=14, color=IFOOD_BLACK)
        plt.ylabel("Número de Eventos", fontsize=14, color=IFOOD_BLACK)
        plt.legend(title="Legenda")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
