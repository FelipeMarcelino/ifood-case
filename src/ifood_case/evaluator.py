
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score

IFOOD_RED = "#EA1D2C"
IFOOD_BLACK = "#3F3E3E"
IFOOD_GRAY_LIGHT = "#E0E0E0"
IFOOD_ORANGE = "#FF7C00"
IFOOD_LIGHT_RED = "#F9868D"


class Evaluator:
    def __init__(self, x_test: pd.DataFrame, y_test: pd.Series):
        self._x_test = x_test
        self._y_test = y_test

    def report(self, y_pred, y_pred_proba):
        metrics = {}

        metrics["roc_auc"] = round(roc_auc_score(self._y_test, y_pred_proba[:, 1]), 2)
        metrics["pr_auc"] = round(average_precision_score(self._y_test, y_pred_proba[:, 1]), 2)

        metrics["taxa_conversao_real_teste"] = round(self._y_test.mean() * 100, 2)
        metrics["taxa_conversao_modelo"] = round(y_pred.mean() * 100, 2)

        report_dict = classification_report(self._y_test, y_pred,  output_dict = True)

        for key, value in report_dict.items():
            if isinstance(value, dict):
                report_dict[key] = {k: round(v, 2) if isinstance(v, float) else v for k, v in value.items()}
            elif isinstance(value, float):
                 report_dict[key] = round(value, 2)

        metrics["classification_report"] =  report_dict



        self.__plot_confusion_matrix(y_pred)

        return metrics

    def __plot_confusion_matrix(self, y_pred):
        cm = confusion_matrix(self._y_test, y_pred)

        group_names = ["Verdadeiro Negativo", "Falso Positivo", "Falso Negativo", "Verdadeiro Positivo"]
        group_counts = [f"{value:0.0f}" for value in cm.flatten()]
        group_percentages = [f"{value:.2%}" for value in cm.flatten() / np.sum(cm)]

        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)

        cmap = sns.light_palette(IFOOD_RED, as_cmap=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=labels, fmt="", cmap=cmap, cbar=False)

        plt.title("Matriz de Confusão", fontsize=16)
        plt.ylabel("Classe Verdadeira", fontsize=12)
        plt.xlabel("Classe Predita", fontsize=12)
        plt.show()

    def calculate_financial_uplift(self, y_pred: np.ndarray, avg_conversion_value: float, offer_cost: float) -> dict[str, float]:
        """Calcula o resultado financeiro da estratégia do modelo em comparação com um baseline.

        Parameters
        ----------
        y_pred : np.ndarray
            As predições de CLASSE (0 ou 1) feitas pelo modelo no conjunto de teste.
        avg_conversion_value : float
            O valor monetário médio estimado de uma conversão bem-sucedida.
        offer_cost : float
            O custo estimado para enviar uma única oferta.

        Returns
        -------
        dict
            Um dicionário com o resultado financeiro do modelo, do baseline e o uplift.

        """
        cm = confusion_matrix(self._y_test, y_pred)
        tn, fp, fn, tp = cm.flatten()

        model_profit = float(round((tp * (avg_conversion_value - offer_cost)) - (fp * offer_cost),2))

        total_customers_in_test = len(self._y_test)
        actual_successes_in_test = self._y_test.sum()

        baseline_profit = float(round((actual_successes_in_test * (avg_conversion_value - offer_cost)) -
            (total_customers_in_test * offer_cost),2))

        uplift = float(round(model_profit - baseline_profit))

        financial_report = {
            "lucro_com_modelo": model_profit,
            "lucro_baseline_(enviar_para_todos)": baseline_profit,
            "uplift_financeiro_(dinheiro_a_mais)": uplift,
            "roi_do_modelo_sobre_baseline (%)": float(round((uplift / abs(baseline_profit)) * 100 if baseline_profit !=
                                                      0 else float("inf"), 2)),
        }

        return financial_report
