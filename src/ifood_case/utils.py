"""Utility functions used across the project for tasks such as column type
separation and model evaluation post-processing.
"""
import logging
import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame

from ifood_case.evaluator import Evaluator

logger = logging.getLogger(__name__)


def get_column_types(
    df: DataFrame, exclude_cols: List[str] = None,
) -> tuple[list[str],list[str]]:
    """Separates the columns of a PySpark DataFrame into numerical and
    categorical lists based on their data types.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The input PySpark DataFrame to analyze.
    exclude_cols : List[str], optional
        A list of column names to exclude from the final lists, such as
        identifiers or the target variable. By default, it excludes a predefined
        set of common identifiers.

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple containing two lists: (numerical_columns, categorical_columns).

    """
    logger.info("Separating columns by numerical and categorical types...")
    if exclude_cols is None:
        exclude_cols = ["account_id", "offer_id", "time_received", "target", "id"]

    numerical_types = ["int", "bigint", "float", "double", "decimal", "short", "byte"]
    categorical_types = ["string", "boolean"]

    numerical_columns = []
    categorical_columns = []

    for column_name, data_type in df.dtypes:
        if column_name in exclude_cols:
            continue

        if data_type in numerical_types:
            numerical_columns.append(column_name)
        elif data_type in categorical_types:
            categorical_columns.append(column_name)
        else:
            logger.warning(f"Column '{column_name}' with type '{data_type}' was not classified.")

    logger.info(f"Found {len(numerical_columns)} numerical and {len(categorical_columns)} categorical columns.")
    return numerical_columns, categorical_columns


def find_optimal_threshold(
    evaluator: Evaluator,
    y_pred_proba: np.ndarray,
    avg_conversion_value: float,
    offer_cost: float,
) -> pd.DataFrame:
    """Tests various probability thresholds to find the one that maximizes
    the financial uplift.

    This function iterates through a range of thresholds, calculates the
    financial outcome for each, and identifies the threshold that yields the
    highest profit compared to a baseline.

    Parameters
    ----------
    evaluator : Evaluator
        An instantiated Evaluator object containing the true test labels (y_test).
    y_pred_proba : np.ndarray
        A 2D numpy array with predicted probabilities from the model.
        Expected shape is (n_samples, 2).
    avg_conversion_value : float
        The estimated average monetary value of a successful conversion.
    offer_cost : float
        The estimated cost to send a single offer.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame detailing the profit and uplift for each tested
        threshold, sorted by uplift.

    """
    logger.info("Starting search for optimal financial threshold...")
    thresholds = np.linspace(0.05, 0.95, 19)  # Test thresholds from 5% to 95%
    results = []

    y_proba_positive_class = y_pred_proba[:, 1]

    for threshold in thresholds:
        y_pred_temp = (y_proba_positive_class >= threshold).astype(int)

        financials = evaluator.calculate_financial_uplift(
            y_pred=y_pred_temp,
            avg_conversion_value=avg_conversion_value,
            offer_cost=offer_cost,
        )

        results.append(
            {
                "threshold": threshold,
                "model_profit": financials["model_profit"],
                "financial_uplift": financials["financial_uplift_(extra_profit)"],
            },
        )

    results_df = pd.DataFrame(results).sort_values(
        by="financial_uplift", ascending=False,
    )
    best_threshold_row = results_df.iloc[0]

    logger.info("--- Threshold vs. Uplift Analysis ---")
    logger.info(f"\n{results_df.to_string()}")

    logger.info("--- Best Threshold Found ---")
    logger.info(f"\n{best_threshold_row}")

    return results_df

def plot_correlation_matrix(df: DataFrame, numerical_cols: List[str]) -> None:
    """Calculates and plots the correlation matrix for the numerical columns
    of a PySpark DataFrame.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The input PySpark DataFrame containing the data.
    numerical_cols : List[str]
        A list of the names of the numerical columns to be included in the
        correlation analysis.

    Returns
    -------
    None
        This function displays a matplotlib plot and does not return any value.

    """
    logger.info("Calculating correlation matrix for numerical features...")
    assembler = VectorAssembler(
        inputCols=numerical_cols, outputCol="features_vector", handleInvalid="skip",
    )
    vector_df = assembler.transform(df).select("features_vector")

    corr_matrix_spark = Correlation.corr(vector_df, "features_vector").head()

    if corr_matrix_spark:
        corr_matrix_array = corr_matrix_spark[0].toArray()

        corr_matrix_pd = pd.DataFrame(
            corr_matrix_array, columns=numerical_cols, index=numerical_cols,
        )

        logger.info("Plotting correlation heatmap...")
        plt.figure(figsize=(16, 12))
        sns.heatmap(
            corr_matrix_pd,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=.5,
        )
        plt.title("Correlation Matrix of Numerical Features", fontsize=18)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    else:
        logger.error("Could not calculate the correlation matrix")

def create_cyclical_features_pandas(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Creates cyclical sine/cosine features from a date column in a Pandas DataFrame.

    This is a utility function to ensure the same feature logic is applied
    during both training and prediction.

    Parameters
    ----------
    df : pd.DataFrame
        The input Pandas DataFrame.
    date_column : str
        The name of the column containing the date information (e.g., 'registered_on').
        The date format is expected to be YYYYMMDD.

    Returns
    -------
    pd.DataFrame
        The DataFrame enriched with cyclical date features (month_sin, month_cos, etc.).

    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # 1. Convert the column to datetime objects
    # The format %Y%m%d matches integers like 20170212
    date_series = pd.to_datetime(df_copy[date_column], format="%Y%m%d")

    # 2. Extract cyclical components
    # Pandas dayofweek: Monday=0, Sunday=6. Add 1 to match PySpark's Sunday=1.
    dayofweek = date_series.dt.dayofweek + 1
    month = date_series.dt.month

    # 3. Apply sine/cosine transformation
    df_copy["month_sin"] = np.sin(2 * math.pi * month / 12)
    df_copy["month_cos"] = np.cos(2 * math.pi * month / 12)
    df_copy["dayofweek_sin"] = np.sin(2 * math.pi * dayofweek / 7)
    df_copy["dayofweek_cos"] = np.cos(2 * math.pi * dayofweek / 7)

    return df_copy
