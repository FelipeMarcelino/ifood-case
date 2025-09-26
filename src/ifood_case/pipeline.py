"""This module defines the main ML pipeline for the iFood case study.

It orchestrates the entire workflow from data loading and feature engineering
to model training, evaluation, and prediction.
"""
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from ifood_case.evaluator import Evaluator
from ifood_case.feature_engineering import FeatureEngineering
from ifood_case.model_trainer import LGBMTrainer, ModelTrainer

# Get a logger for this module
logger = logging.getLogger(__name__)



class TrainingPipeline:
    """Encapsulates the end-to-end model training, evaluation, and prediction pipeline.
    """

    def __init__(self, spark: SparkSession, data_path: str = "data/raw"):
        """Initializes the TrainingPipeline.

        Parameters
        ----------
        spark : pyspark.sql.SparkSession
            The active SparkSession.
        data_path : str, optional
            The path to the raw data directory, by default "data/raw".

        """
        self.spark = spark
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.data_path = self.project_root / data_path
        self.feature_engineer: FeatureEngineering = None
        self.model_trainer: ModelTrainer = None
        self.evaluator: Evaluator = None
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []

    def _load_data(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Loads the raw datasets from the specified path."""
        logger.info(f"Loading raw data from: {self.data_path}")
        offers = self.spark.read.json(os.path.join(self.data_path, "offers.json"))
        transactions = self.spark.read.json(os.path.join(self.data_path, "transactions.json"))
        profile = self.spark.read.json(os.path.join(self.data_path, "profile.json"))
        return offers, transactions, profile

    def train(self)  -> None:
        """Executes the full training pipeline: loads data, engineers features,
        trains the model, and saves the trained artifacts.

        Parameters
        ----------
        model_output_path : str, optional
            The directory path to save the trained model and calibrator,
            by default "models/".

        """
        logger.info("--- Starting Model Training Pipeline ---")

        # 1. Load Data
        offers, transactions, profile = self._load_data()

        # 2. Feature Engineering
        self.feature_engineer = FeatureEngineering(offers, transactions, profile)
        df_features, self.numerical_cols, self.categorical_cols = self.feature_engineer.transform()

        # Cache for performance
        df_features.cache()
        logger.info(f"Feature engineering complete. DataFrame has {df_features.count()} rows.")

        # 3. Model Training and Calibration
        self.model_trainer = LGBMTrainer(
            df=df_features,
            numerical_columns=self.numerical_cols,
            categorical_columns=self.categorical_cols,
            target="target",
        )
        x_train, x_test, y_train, y_test = self.model_trainer.train()

        # 4. Save the trained model and calibrator artifacts
        model_output_path = self.project_root / "models"
        model_output_path.mkdir(parents=True, exist_ok=True)

        model_version = str(uuid.uuid4())
        model_file = f"lgbm_pipeline_{model_version}.joblib"
        calibrator_file = f"isotonic_calibrator_{model_version}.joblib"

        model_path = model_output_path / model_file
        calibrator_path = model_output_path / calibrator_file

        joblib.dump(self.model_trainer._estimator, model_path)
        joblib.dump(self.model_trainer._calibrator, calibrator_path)

        logger.info(f"Trained model pipeline saved to: {model_path}")
        logger.info(f"Trained calibrator saved to: {calibrator_path}")

        logger.info("--- Model Training Pipeline Finished Successfully ---")

        # Note: You can optionally run the evaluation right after training
        self.evaluate(x_test=x_test, y_test=y_test, model_version=model_version)


    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series, model_version: str) -> Dict[str, Any]:
        """Evaluates the trained model on a test set.

        Parameters
        ----------
        x_test : pd.DataFrame
            The test set features.
        y_test : pd.Series
            The test set true labels.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing all calculated performance metrics.

        """
        if not self.model_trainer or not self.model_trainer.estimator:
            raise RuntimeError("Model is not trained yet. Please run the train() method first.")

        logger.info("--- Starting Model Evaluation ---")
        self.evaluator = Evaluator(x_test=x_test, y_test=y_test)

        # Get predictions and probabilities
        predictions = self.model_trainer.predict(x_test)
        probabilities = self.model_trainer.predict_proba(x_test)

        # Generate a full report
        metrics = self.evaluator.report(y_pred=predictions, y_pred_proba=probabilities)

        json_output_path = self.project_root / "models" / f"evaluation_metrics_{model_version}.json"
        with open(json_output_path) as f:
            json.dump(metrics, f, indent=4)

        logger.info("--- Model Evaluation Finished ---")
        return metrics

    @staticmethod
    def predict(input_data: pd.DataFrame, model_id: str) -> np.ndarray:
        """Loads a pre-trained model and makes predictions on new data.
        This is a static method so it can be called without instantiating the pipeline.

        Parameters
        ----------
        input_data : pd.DataFrame
            A pandas DataFrame with the same features used for training.
        model_path : str, optional
            Path to the saved model pipeline file.
        calibrator_path : str, optional
            Path to the saved calibrator file.

        Returns
        -------
        np.ndarray
            A 2D numpy array of calibrated probabilities for each class [prob_0, prob_1].

        """
        project_root = Path(__file__).resolve().parent.parent.parent
        model_path = project_root / "models" / f"lgbm_pipeline_{model_id}.joblib"
        calibrator_path = project_root / "models" / f"isotonic_calibrator_{model_id}.joblib"

        logger.info(f"Loading model from {model_path} for prediction...")
        if not os.path.exists(model_path) or not os.path.exists(calibrator_path):
            raise FileNotFoundError("Model or calibrator file not found. Please train the model first.")

        model = joblib.load(model_path)
        calibrator = joblib.load(calibrator_path)

        logger.info("Making predictions on new data...")
        uncalibrated_probs = model.predict_proba(input_data)[:, 1]
        calibrated_probs_pos = calibrator.predict(uncalibrated_probs)

        # Return probabilities in the standard scikit-learn format
        final_probabilities = np.vstack([1 - calibrated_probs_pos, calibrated_probs_pos]).T
        logger.info("Prediction complete.")

        return final_probabilities
