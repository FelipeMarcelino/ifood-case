import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, concat_ws


class DataVisualizer:
    def __init__(self, offers: DataFrame, transactions: DataFrame, profile: DataFrame):
        self.offers = offers
        self.transactions = transactions
        self.profile = profile


    def plot_countplot_channels(self):
        channels = self.offers \
        .withColumn("channels_str",concat_ws(", ",col("channels"))) \
        .select(col("channels_str")).toPandas()

        plt.figure(figsize=(10,6))
        sns.countplot(data=channels, x="channels_str")
        plt.title("Channels")
        plt.show()
