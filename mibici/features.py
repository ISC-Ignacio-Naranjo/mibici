from pathlib import Path
import pandas as pd
import os
import mlflow
from loguru import logger
import typer

from mibici.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "mibici_cleaned.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "mibici_features.parquet",
):
    logger.info("Iniciando la generación de características (Feature Engineering)...")
    
    if not os.path.exists(input_path):
        logger.error(f"No se encontró el archivo: {input_path}")
        return
        
    df = pd.read_parquet(input_path)
    logger.info(f"Dataset original cargado: {df.shape[0]} filas.")

    # Asegurar formato datetime
    df['inicio_del_viaje'] = pd.to_datetime(df['inicio_del_viaje'], errors='coerce')
    
    # --- CAMBIO 1: Limitar a los últimos 12 meses ---
    logger.info("Limitando el dataset de modelado a los últimos 12 meses...")
    max_date = df['inicio_del_viaje'].max()
    start_date = max_date - pd.DateOffset(months=12)
    df = df[df['inicio_del_viaje'] >= start_date].copy()
    logger.info(f"Registros en ventana de 12 meses: {df.shape[0]} filas.")

    # --- CAMBIO 2: Inyección de ceros para horas sin demanda ---
    logger.info("Construyendo periodos activos e inyectando horas con 0 demanda...")
    df['fecha_hora'] = df['inicio_del_viaje'].dt.floor('h')
    
    hourly_demand = df.groupby(['origen_id', 'fecha_hora']).size().rename('demanda')
    active_periods = df.groupby('origen_id')['fecha_hora'].agg(['min', 'max'])
    
    # Construir todas las combinaciones posibles de estación x hora en su periodo activo
    full_index = pd.MultiIndex.from_tuples(
        [
            (station_id, hour)
            for station_id, row in active_periods.iterrows()
            for hour in pd.date_range(row['min'], row['max'], freq='h')
        ],
        names=['origen_id', 'fecha_hora']
    )
    
    # Completar con 0 donde no hubo viajes
    df_grouped = hourly_demand.reindex(full_index, fill_value=0).reset_index()
    
    # Recuperar variables de fecha y hora
    df_grouped['fecha'] = df_grouped['fecha_hora'].dt.date
    df_grouped['hora'] = df_grouped['fecha_hora'].dt.hour
    df_grouped = df_grouped.drop(columns=['fecha_hora'])

    # --- CAMBIO 3: Variables temporales (y corrección day_of_week) ---
    logger.info("Extrayendo variables temporales...")
    df_grouped['fecha'] = pd.to_datetime(df_grouped['fecha'])
    df_grouped['dia_semana'] = df_grouped['fecha'].dt.day_of_week
    df_grouped['mes'] = df_grouped['fecha'].dt.month
    df_grouped['es_fin_de_semana'] = df_grouped['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)

    df_grouped = df_grouped.sort_values(by=['fecha', 'hora', 'origen_id'])
    logger.info(f"Dataset procesado listo. Tamaño final: {df_grouped.shape}")

    # Guardar archivo final
    logger.info(f"Guardando dataset en {output_path}...")
    os.makedirs(output_path.parent, exist_ok=True)
    df_grouped.to_parquet(output_path, index=False)
    
    # --- CAMBIO 4: Registro óptimo en MLflow (metadata en vez de artefacto pesado) ---
    logger.info("Registrando en MLflow...")
    mlflow.set_experiment("MiBici_Feature_Engineering")
    
    dataset = mlflow.data.from_pandas(
        df_grouped,
        source=str(output_path),
        name="mibici_processed"
    )
    
    with mlflow.start_run(run_name="demanda_horaria_v2"):
        mlflow.log_param("input_file", str(input_path))
        mlflow.log_param("groupby_cols", "origen_id, fecha, hora")
        mlflow.log_metric("total_filas_procesadas", df_grouped.shape[0])
        mlflow.log_input(dataset, context="training")
        
    logger.success("Generación de características completada con éxito.")

if __name__ == "__main__":
    app()
