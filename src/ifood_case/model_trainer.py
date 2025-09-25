from abc import ABC, abstractmethod
from typing import Optional

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

optuna.logging.disable_default_handler()




class ModelTrainer(ABC):
    def __init__(self, df: DataFrame, numerical_columns: list[str], categorical_columns: list[str], target: str):
        self._df = df
        self._numerical_columns = numerical_columns
        self._categorical_columns = categorical_columns
        self._target = target
        self._best_params: Optional[dict[str, float]] = None
        self._best_score: Optional[float] = None
        self._estimator: Optional[Pipeline] = None
        self._calibrator: Optional[IsotonicRegression] = None

    @property
    def best_params(self):
        if self._best_params is None:
            raise RuntimeError("Model not tuned yet. Call train() first")
        return self._best_params

    @property
    def best_score(self):
        if self._best_score is None:
            raise RuntimeError("Model not tuned yet. Call train() first")
        return self._best_score

    @property
    def estimator(self):
        if self._estimator is None:
            raise RuntimeError("Model not tuned yet. Call train() first")
        return self._estimator


    def predict(self, df: pd.DataFrame, threshold: float = 0.5):
        calibrated_probs_pos = self.predict_proba(df)[:, 1]
        return (calibrated_probs_pos >= threshold).astype(int)


    def predict_proba(self, df: pd.DataFrame):
        if self._estimator is None or self._calibrator is None:
            raise RuntimeError("Model or calibrator has not been trained. Call train() first.")

        uncalibrated_probs = self._estimator.predict_proba(df)[:, 1]

        calibrated_probs_pos = self._calibrator.predict(uncalibrated_probs)

        return np.vstack([1 - calibrated_probs_pos, calibrated_probs_pos]).T

    def train(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        x_train, x_test, x_calib, y_train, y_test, y_calib = self._split_data()
        pipeline = self._create_pipeline()
        optuna_search = self._tune_model(pipeline, x_train, y_train)

        self._best_params = optuna_search.best_params_
        self._best_score = optuna_search.best_score_
        self._estimator = optuna_search.best_estimator_

        uncalibrated_probs = self._estimator.predict_proba(x_calib)[:, 1]

        self._calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        self._calibrator.fit(uncalibrated_probs, y_calib)

        return x_train, x_test, y_train, y_test

    def _split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        window_spec = Window.partitionBy("account_id")

        df_with_max_time = self._df.withColumn(
            "max_time_received",
            F.max("time_received").over(window_spec),
        )

        test_df = df_with_max_time.filter(F.col("time_received") == F.col("max_time_received")).drop(
            "max_time_received",
        )

        train_df = df_with_max_time.filter(F.col("time_received") < F.col("max_time_received")).drop(
            "max_time_received",
        )

        train_df = train_df.toPandas()
        test_df = test_df.toPandas()

        x_train = train_df[self._numerical_columns + self._categorical_columns]
        x_test = test_df[self._numerical_columns + self._categorical_columns]

        y_train = train_df[self._target]
        y_test = test_df[self._target]

        x_train, x_calib, y_train, y_calib = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42, stratify=y_train,
        )

        return x_train, x_test, x_calib, y_train, y_test, y_calib

    def _create_pipeline(self) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", StandardScaler(), self._numerical_columns),
                ("categorical", OneHotEncoder(), self._categorical_columns),
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
    def _tune_model(self, pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series):
        pass


class LGBMTrainer(ModelTrainer):
    def __init__(self, df: DataFrame, numerical_columns: list[str], categorical_columns: list[str], target):
        super().__init__(df, numerical_columns, categorical_columns, target)

        self._model = lgb.LGBMClassifier(random_state=42)

    def _create_pipeline(self) -> Pipeline:
        pipeline = super()._create_pipeline()

        pipeline.steps.append(("classifier", self._model))

        return pipeline

    def _tune_model(self, pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series):
        param_distributions = {
            "classifier__n_estimators": IntDistribution(100, 1000),
            "classifier__learning_rate": FloatDistribution(0.01, 0.3),
            "classifier__num_leaves": IntDistribution(20, 300),
            "classifier__max_depth": IntDistribution(3, 12),
            "classifier__reg_alpha": FloatDistribution(0.0, 1.0),  # L1 regularization
            "classifier__reg_lambda": FloatDistribution(0.0, 1.0),  # L2 regularization
        }

        pipeline.set_params(classifier__verbose=-1)

        optuna_search = OptunaSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_trials=50,
            cv=5,
            scoring="average_precision",
            random_state=42,
            verbose=-1,
        )

        optuna_search.fit(x_train, y_train)

        return optuna_search
