from pathlib import Path
import pandas as pd
import glob
import os
import mlflow
from loguru import logger
import typer

from mibici.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    # --- REPLACE DEFAULT PATHS AS APPROPRIATE ---
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = "data/interim/mibici_cleaned.parquet",
    # --------------------------------------------
):
    logger.info("Iniciando el procesamiento del dataset...")
    
    # 1. Cargar datos
    logger.info("Cargando archivos CSV desde raw...")
    file_pattern = os.path.join(input_path, "datos_abiertos_*.csv")
    all_files = glob.glob(file_pattern)
    
    if not all_files:
         logger.error(f"No se encontraron archivos en {input_path}")
         return
         
    df_list = [pd.read_csv(f, encoding='latin1') for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Datos cargados: {df.shape[0]} filas.")

    # 2. Limpieza de columnas
    logger.info("Limpiando columnas y ajustando formatos de fecha...")
    df['anio_nacimiento'] = (
        df.get('Año_de_nacimiento', pd.Series(dtype='float64'))
        .combine_first(df.get('AÃ±o_de_nacimiento', pd.Series(dtype='float64')))
        .combine_first(df.get('A}äe_nacimiento', pd.Series(dtype='float64')))
    )
    cols_to_drop = ['Año_de_nacimiento', 'AÃ±o_de_nacimiento', 'A}äe_nacimiento']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df['Inicio_del_viaje'] = pd.to_datetime(df['Inicio_del_viaje'], format='mixed', errors='coerce')
    df['Fin_del_viaje'] = pd.to_datetime(df['Fin_del_viaje'], format='mixed', errors='coerce')
    df.columns = [col.lower() for col in df.columns]

    # 3. Filtros lógicos
    logger.info("Aplicando filtros lógicos y eliminando nulos...")
    df['genero'] = df['genero'].fillna('Desconocido')
    df = df.dropna(subset=['anio_nacimiento'])
    df = df[df['fin_del_viaje'] >= df['inicio_del_viaje']]
    df = df[df['anio_nacimiento'] >= 1930]
    logger.info(f"Datos después de limpieza: {df.shape[0]} filas.")

    # 4. Guardar archivo
    logger.info(f"Guardando dataset limpio en {output_path}...")
    # Asegurar que el directorio exista
    os.makedirs(output_path.parent, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    # 5. MLflow
    logger.info("Registrando en MLflow...")
    mlflow.set_experiment("MiBici_Data_Preparation")
    with mlflow.start_run(run_name="data_cleaning_v1"):
        mlflow.log_param("raw_data_path", str(input_path))
        mlflow.log_artifact(str(output_path), artifact_path="datasets")
        
    logger.success("Procesamiento del dataset completado.")

if __name__ == "__main__":
    app()