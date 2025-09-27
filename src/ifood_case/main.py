import argparse
import logging

import pandas as pd
from pyspark.sql import SparkSession

from ifood_case.logger_config import setup_logging
from ifood_case.pipeline import TrainingPipeline


def main():
    parser = argparse.ArgumentParser(description="Main pipeline for the iFood case study.")

    # Create a mutually exclusive group: you can either train OR predict, not both.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train",
        action="store_true",
        help="Run the full model training, tuning, and calibration pipeline.",
    )
    group.add_argument(
        "--predict",
        type=str,
        metavar="PATH_TO_DATA",
        help="Run prediction on new data. Requires the path to a CSV file.",
    )

    # Add optional arguments for prediction to specify model files
    parser.add_argument(
        "--model-id",
        type=str,
        help="model id used to load the model",
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger()

    if args.train:
        logger.info("Starting execution in --train mode")

        spark = (
            SparkSession.builder.appName("IfoodTrainingRun")
            .config("spark.driver.memory", "4g")  # Example: 4 gigabytes for the driver
            .config("spark.executor.memory", "8g")  # Example: 8 gigabytes for each executor
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")

        pipeline = TrainingPipeline(spark)
        pipeline.train()

        spark.stop()
        logger.info("Training pipeline finished successfully")

    elif args.predict:
        logger.info(f"Starting execution in --predict mode for file: {args.predict}")

        if not args.model_id:
            logger.error("--model-id is required to predict")

        try:
            input_df = pd.read_csv(args.predict)
            logger.info(f"Loaded {len(input_df)} rows from {args.predict} for prediction.")

            probabilities = TrainingPipeline.predict(
                input_data=input_df,
                model_id=args.model_id,
            )

            input_df["predicted_probability_class_1"] = probabilities[:, 1]
            print("\n--- Prediction Results ---")
            print(input_df.head())

            logger.info("Prediction finished successfully.")

        except FileNotFoundError:
            logger.error(f"Input file not found at: {args.predict}")
        except Exception as e:
            logger.error(f"An error occurred during prediction: {e}", exc_info=True)


if __name__ == "__main__":
    main()
