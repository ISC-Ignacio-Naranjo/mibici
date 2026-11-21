from typing import Any

import mlflow.sklearn
import pandas as pd
import typer
from loguru import logger

from mibici.config import MODEL_URI, PROCESSED_DATA_DIR
from mibici.modeling.train import get_feature_groups


app = typer.Typer()

DEFAULT_MODEL_URI = MODEL_URI


def load_model(model_uri: str = DEFAULT_MODEL_URI) -> Any:
    """Load the selected model from MLflow Model Registry."""
    logger.info(f"Loading model from {model_uri}...")
    model = mlflow.sklearn.load_model(model_uri)
    logger.success("Model loaded successfully.")
    return model


def prepare_input(data: dict | pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare input features expected by the model."""
    _, _, _, feature_cols = get_feature_groups()

    if isinstance(data, dict):
        input_df = pd.DataFrame([data])
    else:
        input_df = data.copy()

    missing_features = [
        feature
        for feature in feature_cols
        if feature not in input_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    return input_df[feature_cols]


def predict_demand(
    model: Any,
    data: dict | pd.DataFrame,
) -> list[float]:
    """Predict hourly demand using the trained MLflow model."""
    input_df = prepare_input(data)
    predictions = model.predict(input_df)

    return [
        float(prediction)
        for prediction in predictions
    ]


@app.command()
def main(
    features_path=PROCESSED_DATA_DIR / "mibici_features.parquet",
    model_uri: str = DEFAULT_MODEL_URI,
    row_index: int = 0,
):
    """Run a sample prediction using the registered champion model."""
    logger.info(f"Reading features from {features_path}...")

    df = pd.read_parquet(features_path)

    if row_index < 0 or row_index >= len(df):
        raise ValueError(
            f"row_index must be between 0 and {len(df) - 1}"
        )

    model = load_model(model_uri)

    _, _, _, feature_cols = get_feature_groups()
    sample = df.iloc[[row_index]][feature_cols]

    predictions = predict_demand(
        model=model,
        data=sample,
    )

    logger.info(
        f"Input: {sample.to_dict(orient='records')[0]}"
    )
    logger.success(
        f"Predicted hourly demand: {predictions[0]}"
    )


if __name__ == "__main__":
    app()