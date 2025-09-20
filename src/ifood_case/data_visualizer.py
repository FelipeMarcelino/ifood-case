"""Módulo responsável por criações das visualizações para insigts sobre os dados."""
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, concat_ws, explode, when

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

    def __init__(self, offers: DataFrame, transactions: DataFrame, profile: DataFrame) -> None:
        """Inicializa o objeto DataVisualizer.

        Parameters
        ----------
        offers : pyspark.sql.DataFrame
            DataFrame com detalhes das ofertas (BOGO, informational, discount).
        transactions : pyspark.sql.DataFrame
            DataFrame com o histórico de transações e interações com ofertas.
        profile : pyspark.sql.DataFrame
            DataFrame com os dados demográficos dos clientes.

        Attributes
        ----------
        offers : pyspark.sql.DataFrame
            Armazena o DataFrame de ofertas fornecido.
        transactions : pyspark.sql.DataFrame
            Armazena o DataFrame de transações fornecido.
        profile : pyspark.sql.DataFrame
            Armazena o DataFrame de perfis fornecido.

        """
        self.offers = offers
        self.transactions = transactions
        self.profile = profile

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
        limit_age = (
            self.profile.select(col("age"), col("credit_card_limit"))
            .withColumn(
                "age_group",
                when(col("age") < 18, "Menor de 18")
                .when((col("age") >= 18) & (col("age") <= 25), "18-25")
                .when((col("age") >= 26) & (col("age") <= 35), "26-35")
                .when((col("age") >= 36) & (col("age") <= 50), "36-50")
                .when(col("age") > 50, "51+")
                .otherwise("Não Informado"),
            )
            .toPandas()
        )


        plt.figure(figsize=(10, 6))
        sns.histplot(data=limit_age, x="credit_card_limit", hue="age_group", palette=IFOOD_PALETTE)
        plt.title("Distribuição de limite de cartão de crédito por idade", color=IFOOD_BLACK)
        plt.xlabel("Limite", color=IFOOD_BLACK)
        plt.ylabel("Contagem", color=IFOOD_BLACK)
        plt.tight_layout()
        plt.show()

    def plot_histogram_gender_credit_card_limit(self) -> None:
        limit_gender = self.profile.select(col("gender"), col("credit_card_limit")) \
                                   .na.fill({"gender": "Nulo"}).toPandas()

        plt.figure(figsize=(10, 6))
        sns.histplot(data=limit_gender, x="credit_card_limit", hue="gender", palette=IFOOD_PALETTE)
        plt.title("Distribuição de limite de cartão de crédito por gênero", color=IFOOD_BLACK)
        plt.xlabel("Limite", color=IFOOD_BLACK)
        plt.ylabel("Contagem", color=IFOOD_BLACK)
        plt.tight_layout()
        plt.show()


