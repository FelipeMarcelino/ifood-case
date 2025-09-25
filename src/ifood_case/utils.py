"""Utils functions used abroad the projects."""
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame

from ifood_case.evaluator import Evaluator


def get_columns_type(df: DataFrame) -> tuple[list[str], list[str]]:
    numerical_types = ["int", "bigint", "float", "double", "decimal", "short", "byte"]
    exclude = ["account_id", "offer_id", "time_received", "target"]

    numerical_columns = []
    categorical_columns = []

    for column_name, data_type in df.dtypes:
        if data_type in numerical_types and column_name not in exclude:
            numerical_columns.append(column_name)
        elif data_type not in numerical_columns and column_name not in exclude:
            categorical_columns.append(column_name)
        else:
            print(f"Column {column_name} is {data_type} or is in exclude list: {exclude}")

    return numerical_columns, categorical_columns

def find_optimal_threshold(evaluator: Evaluator, y_proba: np.ndarray, avg_conversion_value: float, offer_cost: float):
    """Testa vários thresholds de probabilidade para encontrar aquele que maximiza o uplift financeiro.
    """
    thresholds = np.linspace(0.05, 0.95, 19) # Testa thresholds de 0.05 a 0.95
    results = []

    y_proba_positive_class = y_proba[:, 1]

    for threshold in thresholds:
        # Gera predições de classe com base no threshold atual
        y_pred_temp = (y_proba_positive_class >= threshold).astype(int)

        # Calcula o resultado financeiro para este threshold
        financials = evaluator.calculate_financial_uplift(
            y_pred=y_pred_temp,
            avg_conversion_value=avg_conversion_value,
            offer_cost=offer_cost,
        )

        results.append({
            "threshold": threshold,
            "model_profit": financials["lucro_com_modelo"],
            "uplift": financials["uplift_financeiro_(dinheiro_a_mais)"],
        })

    # Converte os resultados para um DataFrame Pandas para fácil análise
    results_df = pd.DataFrame(results)

    # Encontra o melhor threshold
    best_threshold_row = results_df.loc[results_df["uplift"].idxmax()]

    print("Análise de Threshold vs. Uplift:")
    print(results_df)

    print("\n--- Melhor Threshold Encontrado ---")
    print(best_threshold_row)

    return results_df
