"""This module provides the Evaluator class, responsible for calculating and
visualizing performance metrics for a binary classification model.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

# --- Brand Color Constants ---
IFOOD_RED = "#EA1D2C"
IFOOD_BLACK = "#3F3E3E"


class Evaluator:
    """A class to evaluate the performance of a binary classification model.

    It calculates and visualizes key metrics such as the confusion matrix,
    ROC AUC curve, PR AUC curve, and financial uplift.
    """

    def __init__(self, x_test: pd.DataFrame, y_test: pd.Series):
        """Initialize the Evaluator with the ground truth labels.

        Parameters
        ----------
        y_test : pd.Series
            The true target values for the test set.
        y_test : pd.DataFrame
            The dataframe containing the features 

        """
        self._y_test = y_test
        self._x_test = x_test

    def report(
        self,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> dict[str, Any]:
        """Generate a comprehensive evaluation report for the model.

        Calculates key metrics, stores them in a dictionary, and calls
        plotting methods to visualize the model's performance.

        Parameters
        ----------
        y_pred : np.ndarray
            The predicted class labels (0 or 1) from the model.
        y_pred_proba : np.ndarray
            The predicted probabilities for the positive class (1) from the model.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing all calculated performance metrics.

        """
        metrics = {}

        # Calculate ROC AUC and PR AUC from probabilities
        metrics["roc_auc"] = round(roc_auc_score(self._y_test, y_pred_proba[:, 1]), 4)
        metrics["pr_auc"] = round(
            average_precision_score(self._y_test, y_pred_proba[:, 1]),
            4,
        )

        # Calculate real vs. model conversion rates
        metrics["actual_conversion_rate_test (%)"] = round(
            self._y_test.mean() * 100,
            2,
        )
        metrics["model_conversion_rate (%)"] = round(y_pred.mean() * 100, 2)

        # Generate and format the classification report dictionary
        report_dict = classification_report(self._y_test, y_pred, output_dict=True)
        for key, value in report_dict.items():
            if isinstance(value, dict):
                report_dict[key] = {k: round(v, 4) if isinstance(v, float) else v for k, v in value.items()}
            elif isinstance(value, float):
                report_dict[key] = round(value, 4)
        metrics["classification_report"] = report_dict

        # Generate plots
        self.__plot_confusion_matrix(y_pred)
        self.__plot_pr_auc_curve(y_pred_proba[:, 1], metrics["pr_auc"])
        self.__plot_roc_auc_curve(y_pred_proba[:, 1], metrics["roc_auc"])

        return metrics

    def __plot_confusion_matrix(self, y_pred: np.ndarray) -> None:
        """Calculate and plot a detailed confusion matrix.

        Parameters
        ----------
        y_pred : np.ndarray
            The predicted class labels (0 or 1) from the model.

        """
        cm = confusion_matrix(self._y_test, y_pred)

        group_names = [
            "True Negative",
            "False Positive",
            "False Negative",
            "True Positive",
        ]
        group_counts = [f"{value:0.0f}" for value in cm.flatten()]
        group_percentages = [f"{value:.2%}" for value in cm.flatten() / np.sum(cm)]

        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)

        cmap = sns.light_palette(IFOOD_RED, as_cmap=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=labels, fmt="", cmap=cmap, cbar=False)

        plt.title("Confusion Matrix", fontsize=16)
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.show()

    def __plot_pr_auc_curve(self, y_proba_positive_class: np.ndarray, pr_auc: float) -> None:
        """Plot the Precision-Recall (PR) curve for the model.

        Parameters
        ----------
        y_proba_positive_class : np.ndarray
            The predicted probabilities for the positive class (1).
        pr_auc : float
            The pre-calculated PR AUC score to display in the legend.

        """
        precision, recall, _ = precision_recall_curve(
            self._y_test,
            y_proba_positive_class,
        )
        no_skill_baseline = self._y_test.sum() / len(self._y_test)

        plt.figure(figsize=(10, 8))
        plt.plot(
            recall,
            precision,
            color=IFOOD_RED,
            lw=2,
            label=f"PR Curve (AUC = {pr_auc:.4f})",
        )
        plt.plot(
            [0, 1],
            [no_skill_baseline, no_skill_baseline],
            color=IFOOD_BLACK,
            lw=2,
            linestyle="--",
            label=f"Baseline ({no_skill_baseline:.2f})",
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall (PR) Curve", fontsize=16)
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.show()

    def __plot_roc_auc_curve(
        self,
        y_proba_positive_class: np.ndarray,
        roc_auc: float,
    ) -> None:
        """Plot the Receiver Operating Characteristic (ROC) curve.

        Parameters
        ----------
        y_proba_positive_class : np.ndarray
            The predicted probabilities for the positive class (1).
        roc_auc : float
            The pre-calculated ROC AUC score to display in the legend.

        """
        fpr, tpr, _ = roc_curve(self._y_test, y_proba_positive_class)

        plt.figure(figsize=(10, 8))
        plt.plot(
            fpr,
            tpr,
            color=IFOOD_RED,
            lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})",
        )
        plt.plot(
            [0, 1],
            [0, 1],
            color=IFOOD_BLACK,
            lw=2,
            linestyle="--",
            label="Random Classifier",
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FPR)", fontsize=12)
        plt.ylabel("True Positive Rate (TPR)", fontsize=12)
        plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.show()

    def calculate_financial_uplift(
        self,
        y_pred: np.ndarray,
        avg_conversion_value: float,
        offer_cost: float,
    ) -> dict[str, float]:
        """Calculate the financial result of the model's strategy
        compared to a baseline of sending the offer to everyone.

        Parameters
        ----------
        y_pred : np.ndarray
            The predicted class labels (0 or 1) for the test set.
        avg_conversion_value : float
            The estimated average monetary value of a successful conversion.
        offer_cost : float
            The estimated cost to send a single offer.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the financial results for the model,
            the baseline, and the calculated uplift.

        """
        cm = confusion_matrix(self._y_test, y_pred)
        _, fp, _, tp = cm.flatten()

        model_profit = float(
            round((tp * (avg_conversion_value - offer_cost)) - (fp * offer_cost), 2),
        )

        total_customers_in_test = len(self._y_test)
        actual_successes_in_test = self._y_test.sum()

        baseline_profit = float(
            round(
                (actual_successes_in_test * (avg_conversion_value - offer_cost))
                - (total_customers_in_test * offer_cost),
                2,
            ),
        )

        uplift = float(round(model_profit - baseline_profit, 2))

        financial_report = {
            "model_profit": model_profit,
            "baseline_profit_(send_to_all)": baseline_profit,
            "financial_uplift_(extra_profit)": uplift,
            "model_roi_over_baseline (%)": float(
                round(
                    (uplift / abs(baseline_profit)) * 100 if baseline_profit != 0 else float("inf"),
                    2,
                ),
            ),
        }

        return financial_report

    def plot_shap_summary(self, model: Pipeline, top_n: int = 15) -> None:
        """Calculates and plots the SHAP summary plot to show feature importance
        and impact on the model's output.

        Parameters
        ----------
        model : sklearn.pipeline.Pipeline
            The trained scikit-learn pipeline, which must contain a
            preprocessor step and a tree-based classifier step.
        top_n : int, optional
            The number of top features to display on the plot, by default 15.

        Returns
        -------
        None

        """
        preprocessor = model.named_steps["preprocess"]
        classifier = model.named_steps["classifier"]

        x_test_transformed = preprocessor.transform(self._x_test)

        feature_names = preprocessor.get_feature_names_out()

        x_test_transformed_df = pd.DataFrame(x_test_transformed, columns=feature_names)

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(x_test_transformed_df)

        shap.summary_plot(
            shap_values,
            x_test_transformed_df,
            max_display=top_n,
            show=False,
        )
        plt.title(f"Top {top_n} Features - SHAP Summary Plot", fontsize=16)
        plt.tight_layout()
        plt.show()
