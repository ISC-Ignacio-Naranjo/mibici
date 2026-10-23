from pathlib import Path
import pandas as pd
import os
import mlflow
from loguru import logger
import typer

# Importamos las rutas desde la configuración (Asegúrate de que INTERIM_DATA_DIR exista en config.py)
from mibici.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    # --- RUTAS DE ENTRADA Y SALIDA ---
    input_path: Path = INTERIM_DATA_DIR / "mibici_cleaned.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "mibici_features.parquet",
    # --------------------------------------------
):
    logger.info("Iniciando la generación de características (Feature Engineering)...")
    
    # 1. Cargar datos limpios
    logger.info(f"Cargando dataset intermedio desde {input_path}...")
    if not os.path.exists(input_path):
        logger.error(f"No se encontró el archivo: {input_path}")
        return
        
    df = pd.read_parquet(input_path)
    logger.info(f"Dataset cargado: {df.shape[0]} filas.")

    # 2. Agrupación (De viaje individual a demanda horaria)
    logger.info("Agrupando viajes por estación, fecha y hora (Creando variable 'demanda')...")
    # Asegurarnos de que inicio_del_viaje sea datetime
    df['inicio_del_viaje'] = pd.to_datetime(df['inicio_del_viaje'])
    
    # Extraer la fecha sin hora y la hora para agrupar
    df['fecha'] = df['inicio_del_viaje'].dt.date
    df['hora'] = df['inicio_del_viaje'].dt.hour
    
    # El conteo de filas nos da la 'demanda'
    df_grouped = df.groupby(['origen_id', 'fecha', 'hora']).size().reset_index(name='demanda')

    # 3. Ingeniería de Características Temporales
    logger.info("Extrayendo variables temporales (dia_semana, mes, es_fin_de_semana)...")
    df_grouped['fecha'] = pd.to_datetime(df_grouped['fecha'])
    df_grouped['dia_semana'] = df_grouped['fecha'].dt.dayofweek
    df_grouped['mes'] = df_grouped['fecha'].dt.month
    # 1 si es sábado (5) o domingo (6), 0 si es entre semana
    df_grouped['es_fin_de_semana'] = df_grouped['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)

    # Ordenar lógicamente
    df_grouped = df_grouped.sort_values(by=['fecha', 'hora', 'origen_id'])
    logger.info(f"Dataset procesado listo. Tamaño final: {df_grouped.shape}")

    # 4. Guardar archivo final
    logger.info(f"Guardando dataset procesado en {output_path}...")
    os.makedirs(output_path.parent, exist_ok=True)
    df_grouped.to_parquet(output_path, index=False)
    
    # 5. MLflow: Versionado y Trazabilidad
    logger.info("Registrando la nueva versión en MLflow...")
    mlflow.set_experiment("MiBici_Feature_Engineering")
    with mlflow.start_run(run_name="demanda_horaria_v1"):
        mlflow.log_param("input_file", str(input_path))
        mlflow.log_param("groupby_cols", "origen_id, fecha, hora")
        mlflow.log_metric("total_filas_procesadas", df_grouped.shape[0])
        mlflow.log_artifact(str(output_path), artifact_path="datasets/processed")
        
    logger.success("Generación de características completada con éxito.")

if __name__ == "__main__":
    app()