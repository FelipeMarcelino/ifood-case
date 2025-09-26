"""This module contains the abstract base class ModelTrainer and a concrete
implementation for LightGBM, LGBMTrainer. It orchestrates data splitting,
hyperparameter tuning, model training, and calibration.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.integration import OptunaSearchCV
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Suppress Optuna's informational logs for a cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


class ModelTrainer(ABC):
    """An abstract base class for a model training and tuning pipeline.

    This class defines a standard workflow for splitting data, creating a
    preprocessing pipeline, tuning hyperparameters, training a final model,
    and applying isotonic calibration.
    """

    def __init__(self, df: DataFrame, numerical_columns: List[str], categorical_columns: List[str], target: str):
        """Initializes the ModelTrainer.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The final DataFrame containing all features and the target.
        numerical_columns : List[str]
            A list of names for the numerical feature columns.
        categorical_columns : List[str]
            A list of names for the categorical feature columns.
        target : str
            The name of the target column.

        """
        self._df = df
        self._numerical_columns = numerical_columns
        self._categorical_columns = categorical_columns
        self._target = target
        self._best_params: Optional[Dict[str, float]] = None
        self._best_score: Optional[float] = None
        self._estimator: Optional[Pipeline] = None
        self._calibrator: Optional[IsotonicRegression] = None
        logger.info("ModelTrainer initialized")

    @property
    def best_params(self) -> Optional[Dict[str, float]]:
        """Gets the best hyperparameters found during tuning."""
        if self._best_params is None:
            raise RuntimeError("Model not tuned yet. Call train() first")
        return self._best_params

    @property
    def best_score(self) -> Optional[float]:
        """Gets the best score achieved during tuning."""
        if self._best_score is None:
            raise RuntimeError("Model not tuned yet. Call train() first")
        return self._best_score

    @property
    def estimator(self) -> Optional[Pipeline]:
        """Gets the best trained and calibrated estimator pipeline."""
        if self._estimator is None:
            raise RuntimeError("Model not tuned yet. Call train() first")
        return self._estimator

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Makes class predictions (0 or 1) using the calibrated probabilities.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame of features to make predictions on.
        threshold : float, optional
            The probability threshold to classify as the positive class, by default 0.5.

        Returns
        -------
        np.ndarray
            A numpy array of predicted classes (0 or 1).

        """
        calibrated_probs_pos = self.predict_proba(df)[:, 1]
        return (calibrated_probs_pos >= threshold).astype(int)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Makes calibrated probability predictions.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame of features to make predictions on.

        Returns
        -------
        np.ndarray
            A 2D numpy array of calibrated probabilities for each class [prob_0, prob_1].

        """
        if self._estimator is None or self._calibrator is None:
            raise RuntimeError("Model or calibrator has not been trained. Call train() first.")

        uncalibrated_probs = self._estimator.predict_proba(df)[:, 1]
        calibrated_probs_pos = self._calibrator.predict(uncalibrated_probs)

        return np.vstack([1 - calibrated_probs_pos, calibrated_probs_pos]).T

    def train(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Orchestrates the full training, tuning, and calibration pipeline.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            A tuple containing the training and testing dataframes and series:
            (x_train, x_test, y_train, y_test) for final evaluation.

        """
        logger.info("Starting model training and tuning process...")
        x_train, x_test, x_calib, y_train, y_test, y_calib = self._split_data()

        pipeline = self._create_pipeline()
        optuna_search = self._tune_model(pipeline, x_train, y_train)

        self._best_params = optuna_search.best_params_
        self._best_score = optuna_search.best_score_
        self._estimator = optuna_search.best_estimator_

        logger.info(f"Optuna search complete. Best score (average_precision): {self._best_score:.4f}")
        logger.debug(f"Best parameters found: {self._best_params}")

        logger.info("Starting isotonic calibration step...")
        uncalibrated_probs = self._estimator.predict_proba(x_calib)[:, 1]

        self._calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        self._calibrator.fit(uncalibrated_probs, y_calib)
        logger.info("Calibration complete.")

        return x_train, x_test, y_train, y_test

    def _split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Splits data into train, test, and calibration sets using the last-offer strategy.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
            A tuple containing: (x_train, x_test, x_calib, y_train, y_test, y_calib).

        """
        logger.info("Splitting data into train and test sets based on last customer offer.")
        window_spec = Window.partitionBy("account_id")

        df_with_max_time = self._df.withColumn(
            "max_time_received", F.max("time_received").over(window_spec),
        )

        test_df = df_with_max_time.filter(F.col("time_received") == F.col("max_time_received")).drop(
            "max_time_received",
        )

        train_df = df_with_max_time.filter(F.col("time_received") < F.col("max_time_received")).drop(
            "max_time_received",
        )

        logger.info("Converting Spark DataFrames to Pandas.")
        train_df = train_df.toPandas()
        test_df = test_df.toPandas()

        x_train_full = train_df[self._numerical_columns + self._categorical_columns]
        x_test = test_df[self._numerical_columns + self._categorical_columns]

        y_train_full = train_df[self._target]
        y_test = test_df[self._target]

        logger.info("Splitting training data into a training subset and a calibration set.")
        x_train, x_calib, y_train, y_calib = train_test_split(
            x_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full,
        )
        logger.info(f"Data split: {len(x_train)} train, {len(x_calib)} calibration, {len(x_test)} test samples.")

        return x_train, x_test, x_calib, y_train, y_test, y_calib

    def _create_pipeline(self) -> Pipeline:
        """Creates the base preprocessing pipeline for features.

        Returns
        -------
        sklearn.pipeline.Pipeline
            A scikit-learn pipeline with preprocessing steps.

        """
        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", StandardScaler(), self._numerical_columns),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), self._categorical_columns),
            ],
            remainder="passthrough",
        )
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
            ],
        )
        return pipeline

    @abstractmethod
    def _tune_model(self, pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> OptunaSearchCV:
        """Abstract method for hyperparameter tuning.

        This method must be implemented by subclasses to define the model-specific
        hyperparameter search space and execute the tuning process.

        Parameters
        ----------
        pipeline : sklearn.pipeline.Pipeline
            The preprocessing pipeline to which the model will be added.
        x_train : pd.DataFrame
            The training feature data.
        y_train : pd.Series
            The training target data.

        Returns
        -------
        optuna.integration.OptunaSearchCV
            The fitted OptunaSearchCV object.

        """


class LGBMTrainer(ModelTrainer):
    """A concrete implementation of ModelTrainer for a LightGBM classifier.
    """

    def __init__(self, df: DataFrame, numerical_columns: List[str], categorical_columns: List[str], target: str):
        """Initializes the LGBMTrainer.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The final DataFrame containing all features and the target.
        numerical_columns : List[str]
            A list of names for the numerical feature columns.
        categorical_columns : List[str]
            A list of names for the categorical feature columns.
        target : str
            The name of the target column.

        """
        super().__init__(df, numerical_columns, categorical_columns, target)
        self._model = lgb.LGBMClassifier(random_state=42)

    def _create_pipeline(self) -> Pipeline:
        """Creates the full pipeline by adding the LGBM classifier to the base
        preprocessing pipeline.

        Returns
        -------
        sklearn.pipeline.Pipeline
            The complete scikit-learn pipeline including model.

        """
        pipeline = super()._create_pipeline()
        pipeline.steps.append(("classifier", self._model))
        return pipeline

    def _tune_model(self, pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> OptunaSearchCV:
        """Implements the hyperparameter tuning for the LightGBM model using Optuna.

        Parameters
        ----------
        pipeline : sklearn.pipeline.Pipeline
            The complete preprocessing and model pipeline.
        x_train : pd.DataFrame
            The training feature data.
        y_train : pd.Series
            The training target data.

        Returns
        -------
        optuna.integration.OptunaSearchCV
            The fitted OptunaSearchCV object containing the best model and results.

        """
        param_distributions = {
            "classifier__n_estimators": IntDistribution(100, 1000),
            "classifier__learning_rate": FloatDistribution(0.01, 0.3),
            "classifier__num_leaves": IntDistribution(20, 300),
            "classifier__max_depth": IntDistribution(3, 12),
            "classifier__reg_alpha": FloatDistribution(0.0, 1.0),
            "classifier__reg_lambda": FloatDistribution(0.0, 1.0),
        }

        pipeline.set_params(classifier__verbose=-1)

        optuna_search = OptunaSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_trials=50,
            cv=5,
            scoring="average_precision",
            random_state=42,
            verbose=0, # Set to 0 or higher integer for more verbosity
        )

        logger.info(f"Starting OptunaSearchCV for LightGBM with {optuna_search.n_trials} trials...")
        optuna_search.fit(x_train, y_train)

        return optuna_search
