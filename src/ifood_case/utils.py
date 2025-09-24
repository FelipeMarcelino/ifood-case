"""Utils functions used abroad the projects."""
from pyspark.sql import DataFrame


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
