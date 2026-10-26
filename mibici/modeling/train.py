from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mibici.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "mibici_features.parquet",
    cutoff_date: str = "2026-08-01",
    results_path: Path = Path("reports/baseline_results.csv"),
):
    """Entrena y evalúa baselines para la demanda horaria de MiBici."""

    logger.info(f"Cargando dataset procesado desde {features_path}...")

    if not features_path.exists():
        logger.error(f"No se encontró el dataset: {features_path}")
        raise typer.Exit(code=1)

    df = pd.read_parquet(features_path)

    # Reconstruir fecha_hora si el dataset procesado contiene fecha y hora.
    if "fecha_hora" not in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"])

        df["fecha_hora"] = (
            df["fecha"]
            + pd.to_timedelta(df["hora"], unit="h")
        )

    cutoff = pd.Timestamp(cutoff_date)

    # Separación cronológica para evitar mezclar futuro con pasado.
    train = df[df["fecha_hora"] < cutoff].copy()
    test = df[df["fecha_hora"] >= cutoff].copy()

    feature_cols = [
        "origen_id",
        "hora",
        "dia_semana",
        "mes",
        "es_fin_de_semana",
    ]

    X_train = train[feature_cols]
    y_train = train["demanda"]

    X_test = test[feature_cols]
    y_test = test["demanda"]

    logger.info(
        f"Train: {len(train)} filas | "
        f"Test: {len(test)} filas | "
        f"Cutoff: {cutoff.date()}"
    )

    mlflow.set_experiment("MiBici_Baseline_Modeling")

    results = []

    for strategy in ["mean", "median"]:
        run_name = f"dummy_{strategy}_baseline"

        with mlflow.start_run(run_name=run_name) as run:
            model = DummyRegressor(strategy=strategy)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(
                mean_squared_error(y_test, predictions)
            )

            mlflow.log_param("model_type", "DummyRegressor")
            mlflow.log_param("strategy", strategy)
            mlflow.log_param("target", "demanda")
            mlflow.log_param("split_type", "temporal")
            mlflow.log_param("cutoff_date", cutoff_date)
            mlflow.log_param("features", ",".join(feature_cols))
            mlflow.log_param("dataset", str(features_path))

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("train_rows", len(train))
            mlflow.log_metric("test_rows", len(test))

            mlflow.sklearn.log_model(
                model,
                name="model",
            )

            results.append(
                {
                    "run_id": run.info.run_id,
                    "strategy": strategy,
                    "mae": mae,
                    "rmse": rmse,
                }
            )

            logger.info(
                f"{strategy}: MAE={mae:.4f} | RMSE={rmse:.4f}"
            )

    results_df = pd.DataFrame(results)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)

    logger.success(
        f"Resultados guardados en {results_path}"
    )

    print(results_df.to_string(index=False))


if __name__ == "__main__":
    app()
