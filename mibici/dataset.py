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
    # Buscamos los archivos CSV en las subcarpetas de cada año
    file_pattern = os.path.join(input_path, "**", "datos_abiertos_*.csv")
    all_files = sorted(glob.glob(file_pattern, recursive=True))
    
    if not all_files:
         logger.error(f"No se encontraron archivos en {input_path}")
         return
         
    # Cargamos cada archivo y convertimos sus fechas antes de unirlos
    df_list = []

    for f in all_files:
        df_mes = pd.read_csv(f, encoding='latin1')

        # Identificamos los meses que utilizan el formato día/mes/año
        if Path(f).stem in ("datos_abiertos_2024_09", "datos_abiertos_2024_10"):
            formato_fecha = "%d/%m/%Y %H:%M"
        else:
            formato_fecha = "%Y-%m-%d %H:%M:%S"

        # Convertimos las fechas utilizando el formato correspondiente
        for columna in ("Inicio_del_viaje", "Fin_del_viaje"):
            df_mes[columna] = pd.to_datetime(
                df_mes[columna],
                format=formato_fecha,
                errors="coerce"
            )

        df_list.append(df_mes)

    # Unimos los 24 archivos después de convertir las fechas
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
    df.columns = [col.lower() for col in df.columns]

    # 3. Filtros lógicos
    logger.info("Aplicando filtros lógicos y eliminando nulos...")
    df['genero'] = df['genero'].fillna('Desconocido')
    df = df[df['fin_del_viaje'] >= df['inicio_del_viaje']]
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