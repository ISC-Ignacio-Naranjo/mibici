from pathlib import Path
import pandas as pd
import os
import mlflow
from loguru import logger
import typer

from mibici.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

def load_data(input_path: Path) -> pd.DataFrame:
    """Carga el dataset interim y asegura el formato datetime."""
    if not os.path.exists(input_path):
        logger.error(f"No se encontró el archivo: {input_path}")
        raise FileNotFoundError(f"Archivo no encontrado: {input_path}")
    
    df = pd.read_parquet(input_path)
    logger.info(f"Dataset original cargado: {df.shape[0]} filas.")
    df['inicio_del_viaje'] = pd.to_datetime(df['inicio_del_viaje'], errors='coerce')
    return df

def filter_time_window(df: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    """Limita el dataset a los últimos 'months' meses."""
    logger.info(f"Limitando el dataset a los últimos {months} meses...")
    max_date = df['inicio_del_viaje'].max()
    start_date = max_date - pd.DateOffset(months=months)
    df_filtered = df[df['inicio_del_viaje'] >= start_date].copy()
    logger.info(f"Registros en ventana de {months} meses: {df_filtered.shape[0]} filas.")
    return df_filtered

def inject_zero_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Inyecta ceros en periodos sin demanda respetando las ventanas activas por estación."""
    logger.info("Construyendo periodos activos e inyectando horas con 0 demanda...")
    df['fecha_hora'] = df['inicio_del_viaje'].dt.floor('h')
    
    hourly_demand = df.groupby(['origen_id', 'fecha_hora']).size().rename('demanda')
    active_periods = df.groupby('origen_id')['fecha_hora'].agg(['min', 'max'])
    
    full_index = pd.MultiIndex.from_tuples(
        [
            (station_id, hour)
            for station_id, row in active_periods.iterrows()
            for hour in pd.date_range(row['min'], row['max'], freq='h')
        ],
        names=['origen_id', 'fecha_hora']
    )
    
    df_grouped = hourly_demand.reindex(full_index, fill_value=0).reset_index()
    
    df_grouped['fecha'] = df_grouped['fecha_hora'].dt.date
    df_grouped['hora'] = df_grouped['fecha_hora'].dt.hour
    df_grouped = df_grouped.drop(columns=['fecha_hora'])
    return df_grouped

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae características temporales como día de la semana, mes y fin de semana."""
    logger.info("Extrayendo variables temporales...")
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['dia_semana'] = df['fecha'].dt.day_of_week
    df['mes'] = df['fecha'].dt.month
    df['es_fin_de_semana'] = df['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)
    
    df = df.sort_values(by=['fecha', 'hora', 'origen_id'])
    return df

def generate_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera características históricas de demanda (lags) para predecir sin data leakage."""
    logger.info("Calculando variables históricas (lag features)...")
    # Aseguramos el orden cronológico estricto por estación
    df = df.sort_values(by=['origen_id', 'fecha', 'hora'])
    
    # Lag 1: Demanda de la hora anterior
    df['demanda_lag_1h'] = df.groupby('origen_id')['demanda'].shift(1)
    
    # Lag 24: Demanda a la misma hora del día anterior
    df['demanda_lag_24h'] = df.groupby('origen_id')['demanda'].shift(24)
    
    # Eliminamos las primeras filas que quedan con NaNs debido a los lags
    df = df.dropna(subset=['demanda_lag_1h', 'demanda_lag_24h'])
    
    logger.info(f"Variables de lag generadas. Tamaño resultante: {df.shape[0]} filas.")
    return df

def log_features_to_mlflow(df: pd.DataFrame, input_path: Path, output_path: Path):
    """Registra el dataset procesado y los parámetros de Feature Engineering en MLflow."""
    logger.info("Registrando en MLflow...")
    mlflow.set_experiment("MiBici_Feature_Engineering")
    
    dataset = mlflow.data.from_pandas(
        df,
        source=str(output_path),
        name="mibici_processed_with_lags"
    )
    
    with mlflow.start_run(run_name="demanda_horaria_con_lags"):
        mlflow.log_param("input_file", str(input_path))
        mlflow.log_param("groupby_cols", "origen_id, fecha, hora")
        mlflow.log_metric("total_filas_procesadas", df.shape[0])
        mlflow.log_input(dataset, context="training")


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "mibici_cleaned.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "mibici_features.parquet",
):
    logger.info("Iniciando pipeline modular de Feature Engineering...")
    
    df_raw = load_data(input_path)
    df_window = filter_time_window(df_raw, months=12)
    df_grouped = inject_zero_demand(df_window)
    df_features = extract_time_features(df_grouped)
    
    # Integración del nuevo requisito de la Fase 2 (Variables de demanda histórica)
    df_final = generate_lag_features(df_features)
    
    logger.info(f"Guardando dataset en {output_path}...")
    os.makedirs(output_path.parent, exist_ok=True)
    df_final.to_parquet(output_path, index=False)
    
    log_features_to_mlflow(df_final, input_path, output_path)
    logger.success("Generación de características completada con éxito.")

if __name__ == "__main__":
    app()