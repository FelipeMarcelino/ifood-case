import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, concat_ws, explode


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

    def plot_countplot_channels(self) -> None:
        """Plota a contagem para a coluna 'channels' de duas formas.

        Este método serve como um ponto de entrada que chama dois métodos
        privados para plotar a contagem de combinações de canais e também
        a contagem de canais individuais.

        Returns
        -------
        None

        """
        self.__plot_countplot_channels_joined()
        self.__plot_count_channels_separated()

    def __plot_countplot_channels_joined(self) -> None:
        """Plota a contagem de combinações de canais (ex: 'email, web')."""
        channels = self.offers \
            .withColumn("channels_str", concat_ws(", ", col("channels"))) \
            .select(col("channels_str")).toPandas()

        plt.figure(figsize=(10, 6))
        sns.countplot(data=channels, x="channels_str")
        plt.title("Contagem por Combinação de Canais")
        plt.xlabel("Combinação de Canais")
        plt.ylabel("Contagem")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def __plot_count_channels_separated(self) -> None:
        """Plota a contagem de cada canal de oferta individualmente."""
        channels_exploded = self.offers.withColumn("channels_exp", explode(col("channels")))
        channels_counts = channels_exploded.groupBy("channels_exp").count().orderBy("count", ascending=False)
        channels_pd = channels_counts.toPandas()

        plt.figure(figsize=(10, 6))
        sns.barplot(data=channels_pd, x="channels_exp", y="count")
        plt.title("Contagem por Canal Individual")
        plt.xlabel("Canal")
        plt.ylabel("Contagem")
        plt.show()
