"""Utility functions used across the project for tasks such as column type
separation and model evaluation post-processing.
"""
from typing import List

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame

from ifood_case.evaluator import Evaluator


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
            print(
                f"Column '{column_name}' with type '{data_type}' was not classified.",
            )

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

    print("--- Threshold vs. Uplift Analysis ---")
    print(results_df.to_string())

    print("\n--- Best Threshold Found ---")
    print(best_threshold_row)

    return results_df
