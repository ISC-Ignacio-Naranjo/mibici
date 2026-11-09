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

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.decomposition import TruncatedSVD

app = typer.Typer()

def load_data(features_path: Path) -> pd.DataFrame:
    """Carga el dataset procesado utilizado para entrenamiento."""

    logger.info(
        f"Cargando dataset procesado desde {features_path}..."
    )

    if not features_path.exists():
        logger.error(
            f"No se encontró el dataset: {features_path}"
        )
        raise typer.Exit(code=1)

    return pd.read_parquet(features_path)

def prepare_temporal_split(
        df: pd.DataFrame,
        cutoff_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Prepara la fecha temporal y separa entrenamiento y prueba."""

    if "fecha_hora" not in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"])

        df["fecha_hora"] = (
            df["fecha"]
            + pd.to_timedelta(df["hora"], unit="h")
        )

    cutoff = pd.Timestamp(cutoff_date)

    train = (
        df[df["fecha_hora"] < cutoff]
        .sort_values("fecha_hora")
        .copy()
    )

    test = (
        df[df["fecha_hora"] >= cutoff]
        .sort_values("fecha_hora")
        .copy()
    )

    return train, test, cutoff

def get_feature_groups() -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
    ]:
    """Define los grupos de variables utilizados por el modelo."""

    categorical_features = [
        "origen_id",
        "hora",
        "dia_semana",
        "mes"
    ]

    numeric_features = [
        "demanda_lag_1h",
        "demanda_lag_24h",
    ]

    binary_features = [
        "es_fin_de_semana",
    ]

    features_cols = (
        categorical_features
        + numeric_features
        + binary_features
    )

    return (
        categorical_features,
        numeric_features,
        binary_features,
        features_cols,
    )

def build_preprocessor(
        categorical_features: list[str],
        numeric_features: list[str],
        binary_features: list[str],
) -> ColumnTransformer:
    """Construye el preprocesamiento para las variables del model."""

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
            (
                "binary",
                "passthrough",
                binary_features,
            ),
        ]
    )

def build_ridge_search(
        preprocessor: ColumnTransformer,
) -> GridSearchCV:
    """Construye la búsqueda de hiperparámetros para Ridge."""

    ridge_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge()),
        ]
    )

    param_grid = {
        "model__alpha": [
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
        ]
    }

    time_series_cv = TimeSeriesSplit(n_splits=3)

    return GridSearchCV(
        estimator=ridge_pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=time_series_cv,
        n_jobs=2,
        verbose=2,
    )

def build_ridge_svd_pipeline(
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """Construye Ridge con reducción de dimensionalidad mediante TruncatedSVD."""

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "dimensionality_reduction",
                TruncatedSVD(
                    n_components=50,
                    random_state=42,
                ),
            ),
            (
                "model",
                Ridge(alpha=100.0),
            ),
        ]
    )

def build_sgd_search(
    preprocessor: ColumnTransformer,
) -> GridSearchCV:
    """Construye la búsqueda de hiperparámetros para SGDRegressor."""

    sgd_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                SGDRegressor(
                    max_iter=1000,
                    tol=1e-3,
                    random_state=42,
                ),
            ),
        ]
    )

    param_grid = {
        "model__alpha": [
            0.00001,
            0.0001,
            0.001,
        ]
    }

    time_series_cv = TimeSeriesSplit(n_splits=3)

    return GridSearchCV(
        estimator=sgd_pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=time_series_cv,
        n_jobs=2,
        verbose=2,
    )

def prepare_model_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple [
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Separa variables predictoras y variables objetivo."""

    X_train = train[feature_cols]
    y_train = train["demanda"]

    X_test = test[feature_cols]
    y_test = test["demanda"]

    return X_train, y_train, X_test, y_test

def tune_model(
    grid_search: GridSearchCV,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
) -> Pipeline:
    """Ajusta un modelo y devulve el mejor pipeline encontrado."""

    grid_search.fit(X_train, y_train)

    logger.info(
        f"{model_name} - Mejor alpha: "
        f"{grid_search.best_params_['model__alpha']}"
    )

    logger.info(
        f"{model_name} - Mejor MAE de validación: "
        f"{-grid_search.best_score_:.4f}"
    )

    return grid_search.best_estimator_

def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[float, float]:
    """Evalúa un modelo utilizando MAE y RMSE."""

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )
    return mae, rmse

def log_model_run(
        model: Pipeline,
        model_name: str,
        run_name: str,
        mae: float,
        rmse: float,
        train_rows: int,
        test_rows: int,
        cutoff_date: str,
        feature_cols: list[str],
        best_alpha: float,
        extra_params: dict | None = None,
) -> str:
    """Registra un modelo, sus parámetros y métricas en MLflow."""

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("alpha", best_alpha)
        if extra_params:
            mlflow.log_params(extra_params)
        mlflow.log_param("split_type", "temporal")
        mlflow.log_param("cutoff_date", cutoff_date)
        mlflow.log_param(
            "features",
            ",".join(feature_cols),
        )

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("train_rows", train_rows)
        mlflow.log_metric("test_rows", test_rows)

        # Guardar preprocessing y modelo como un solo artefacto.
        mlflow.sklearn.log_model(
            model,
            name="model",
        )

        logger.info(
            f"{model_name}: MAE={mae:.4f} | RMSE={rmse:.4f}"
        )

        return run.info.run_id

def run_dummy_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    train_rows: int,
    test_rows: int,
    cutoff_date: str,
    feature_cols: list[str],
    features_path: Path,
) -> list[dict]:
    """Entrena y registra los modelos baseline en MLflow."""

    results = []

    for strategy in ["mean", "median"]:
        run_name = f"dummy_{strategy}_baseline"

        with mlflow.start_run(run_name=run_name) as run:
            model = DummyRegressor(strategy=strategy)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            mae = mean_absolute_error(
                y_test,
                predictions,
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions,
                )
            )

            mlflow.log_param(
                "model_type",
                "DummyRegressor",
            )
            mlflow.log_param(
                "strategy",
                strategy,
            )
            mlflow.log_param(
                "target",
                "demanda",
            )
            mlflow.log_param(
                "split_type",
                "temporal",
            )
            mlflow.log_param(
                "cutoff_date",
                cutoff_date,
            )
            mlflow.log_param(
                "features",
                ",".join(feature_cols),
            )
            mlflow.log_param(
                "dataset",
                str(features_path),
            )

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric(
                "train_rows",
                train_rows,
            )
            mlflow.log_metric(
                "test_rows",
                test_rows,
            )

            mlflow.sklearn.log_model(
                model,
                name="model",
            )

            results.append(
                {
                    "model": "DummyRegressor",
                    "configuration": f"strategy={strategy}",
                    "mae": mae,
                    "rmse": rmse,
                }
            )

            logger.info(
                f"{strategy}: "
                f"MAE={mae:.4f} | "
                f"RMSE={rmse:.4f}"
            )

    return results

def save_results(
    results: list[dict],
    results_path: Path,
) -> None:
    """Guarda los resultados de los baselines en un archivo CSV."""

    results_df = pd.DataFrame(results)

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    logger.success(
        f"Resultados guardados en {results_path}"
    )

    print(results_df.to_string(index=False))


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "mibici_features.parquet",
    cutoff_date: str = "2026-08-01",
    results_path: Path = Path("reports/baseline_results.csv"),
):
    """Entrena, ajusta y evalúa modelos para la demanda horaria de MiBici."""

    df = load_data(features_path)

    train, test, cutoff = prepare_temporal_split(
        df,
        cutoff_date,
    )

    (
        categorical_features,
        numeric_features,
        binary_features,
        feature_cols,
    ) = get_feature_groups()

    preprocessor = build_preprocessor(
        categorical_features,
        numeric_features,
        binary_features,
    )

    grid_search = build_ridge_search(preprocessor)

    sgd_search = build_sgd_search(preprocessor)

    ridge_svd_pipeline = build_ridge_svd_pipeline(
        preprocessor,
    )

    X_train, y_train, X_test, y_test = prepare_model_data(
        train,
        test,
        feature_cols,
    )

    best_ridge_pipeline = tune_model(
        grid_search,
        X_train,
        y_train,
        model_name="Ridge",
    )

    best_sgd_pipeline = tune_model(
        sgd_search,
        X_train,
        y_train,
        model_name="SGDRegressor",
    )

    # Entrenar la variante de Ridge con reducción de dimensionalidad.
    ridge_svd_pipeline.fit(
        X_train,
        y_train,
    )

    ridge_mae, ridge_rmse = evaluate_model(
        best_ridge_pipeline,
        X_test,
        y_test,
    )

    sgd_mae, sgd_rmse = evaluate_model(
        best_sgd_pipeline,
        X_test,
        y_test,
    )

    ridge_svd_mae, ridge_svd_rmse = evaluate_model(
        ridge_svd_pipeline,
        X_test,
        y_test,
    )

    logger.info(
        f"Ridge + TruncatedSVD: "
        f"MAE={ridge_svd_mae:.4f} |"
        f"RMSE={ridge_svd_rmse:.4f}"
    )

    # Seleccionar el experimento donde se registrarán todos los modelos.
    mlflow.set_experiment("MiBici_Baseline_Modeling")

    log_model_run(
        model=best_ridge_pipeline,
        model_name="Ridge",
        run_name="ridge_tuned",
        mae=ridge_mae,
        rmse=ridge_rmse,
        train_rows=len(train),
        test_rows=len(test),
        cutoff_date=cutoff_date,
        feature_cols=feature_cols,
        best_alpha=grid_search.best_params_["model__alpha"],
    )

    log_model_run(
        model=best_sgd_pipeline,
        model_name="SGDRegressor",
        run_name="sgd_tuned",
        mae=sgd_mae,
        rmse=sgd_rmse,
        train_rows=len(train),
        test_rows=len(test),
        cutoff_date=cutoff_date,
        feature_cols=feature_cols,
        best_alpha=sgd_search.best_params_["model__alpha"],
        )

    log_model_run(
        model=ridge_svd_pipeline,
        model_name="Ridge+TruncatedSVD",
        run_name="ridge_svd",
        mae=ridge_svd_mae,
        rmse=ridge_svd_rmse,
        train_rows=len(train),
        test_rows=len(test),
        cutoff_date=cutoff_date,
        feature_cols=feature_cols,
        best_alpha=100.0,
        extra_params={
            "n_components": 50,
        },
    )


    logger.info(
        f"Train: {len(train)} filas | "
        f"Test: {len(test)} filas | "
        f"Cutoff: {cutoff.date()}"
    )

    results = run_dummy_baselines(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_rows=len(train),
        test_rows=len(test),
        cutoff_date=cutoff_date,
        feature_cols=feature_cols,
        features_path=features_path,
    )

    results.extend(
        [
            {
                "model": "Ridge",
                "configuration": (
                    f"alpha={grid_search.best_params_['model__alpha']}"
                ),
                "mae": ridge_mae,
                "rmse": ridge_rmse,
            },
            {
                "model": "SGDRegressor",
                "configuration": (
                    f"alpha={sgd_search.best_params_['model__alpha']}"
                ),
                "mae": sgd_mae,
                "rmse": sgd_rmse,
            },
            {
                "model": "Ridge+TruncatedSVD",
                "configuration": "alpha=100.0,n_components=50",
                "mae": ridge_svd_mae,
                "rmse": ridge_svd_rmse,
            },
        ]
    )



    save_results(
        results,
        results_path,
    )


if __name__ == "__main__":
    app()