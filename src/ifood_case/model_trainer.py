from abc import ABC, abstractmethod
from typing import Optional

import lightgbm as lgb
import pandas as pd
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.integration import OptunaSearchCV
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class ModelTrainer(ABC):
    def __init__(self, df: DataFrame, numerical_columns: list[str], categorical_columns: list[str], target: str):
        self._df = df
        self._numerical_columns = numerical_columns
        self._categorical_columns = categorical_columns
        self._target = target
        self._best_params: Optional[dict[str, float]] = None
        self._best_score: Optional[float] = None

    @property
    def best_params(self):
        if self.best_params is None:
            print("Model not tuned yet.")
        return self._best_params

    @property
    def best_score(self):
        if self.best_score is None:
            print("Model not tuned yet.")
        return self._best_score

    def predict(self):
        self.model.predict()

    def predict(self):
        self.model.predict_proba()

    def train(self):
        x_train, x_test, y_train, y_test = self.__split_data()
        pipeline = self._create_pipeline()
        optuna_search = self._tune_model(pipeline, x_train, y_train)

        self._best_params = optuna_search.best_params_
        self._best_score = optuna_search.best_score_

    def _split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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

        x_train = train_df.loc[self._numerical_columns + self._categorical_columns]
        x_test = test_df.loc[self._numerical_columns + self._categorical_columns]

        y_train = train_df.loc[self._target]
        y_test = test_df.loc[self._target]

        return x_train, x_test, y_train, y_test

    def _create_pipeline(self) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", OneHotEncoder(), self._categorical_columns),
            ],
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

        self.model = lgb.LGBMClassifier(random_state=42)

    def _create_pipeline(self) -> Pipeline:
        pipeline = super()._create_pipeline()

        pipeline.steps.append(("classifier"), self.model)

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

        optuna_search = OptunaSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=50,
            cv=3,
            scoring="average_precision",
            random_state=42,
            verbose=1,
        )

        optuna_search.fit(x_train, y_train)

        return optuna_search
