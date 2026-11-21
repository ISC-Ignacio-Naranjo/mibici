from pathlib import Path
import pandas as pd
import glob
import os
import mlflow
from loguru import logger
import typer

from mibici.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


def parse_trip_dates(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Convertimos las fechas de los viajes según el formato del archivo."""
    df = df.copy()

    if Path(filename).stem in (
        "datos_abiertos_2024_09",
        "datos_abiertos_2024_10",
    ):
        formato_fecha = "%d/%m/%Y %H:%M"
    else:
        formato_fecha = "%Y-%m-%d %H:%M:%S"

    for columna in ("Inicio_del_viaje", "Fin_del_viaje"):
        df[columna] = pd.to_datetime(
            df[columna],
            format=formato_fecha,
            errors="coerce",
        )

    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica las variantes del año de nacimiento y normaliza encabezados."""
    df = df.copy()

    variantes = [
        "Año_de_nacimiento",
        "AÃ±o_de_nacimiento",
        "A}äe_nacimiento",
    ]

    presentes = [col for col in variantes if col in df.columns]

    if presentes:
        df["anio_nacimiento"] = df[presentes].bfill(axis=1).iloc[:, 0]
        df = df.drop(columns=presentes)
    else:
        df["anio_nacimiento"] = pd.NA

    df.columns = [col.lower() for col in df.columns]

    
    return df


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica reglas de limpieza conservando viajes útiles."""
    df = df.copy()

    df["genero"] = df["genero"].fillna("Desconocido")

    df = df[
        df["inicio_del_viaje"].notna()
        & df["fin_del_viaje"].notna()
        & (df["fin_del_viaje"] >= df["inicio_del_viaje"])
    ].copy()

    return df



def load_raw_data(input_path: Path) -> pd.DataFrame:
    """Carga los archivos CSV originales y normaliza sus fechas."""

    # Buscar archivos CSV en las carpetas y subcarpetas.
    file_pattern = os.path.join(str(input_path), "**", "datos_abiertos_*.csv")
    all_files = sorted(glob.glob(file_pattern, recursive=True))

    # Verificar que existan archivos para procesar.
    if not all_files:
        raise FileNotFoundError(
            f"No se encontraron archivos CSV en {input_path}"
        )

    # Leer los archivos mensuales y convertir sus fechas.
    df_list = []

    for archivo in all_files:
        df_mes = pd.read_csv(archivo, encoding="latin1")

        # Evitar interpretaciones incorrectas de día y mes.
        df_mes = parse_trip_dates(df_mes, archivo)

        # Conservar los registros de cada archivo.
        df_list.append(df_mes)

    # Combinar todos los registros en un único DataFrame.
    return pd.concat(df_list, ignore_index=True)

def load_raw_snapshot(input_path: Path) -> pd.DataFrame:
    """Lee los CSV originales sin aplicar transformaciones de limpieza."""

    # Localizar los archivos originales en las subcarpetas.
    file_pattern = os.path.join(
        str(input_path), "**", "datos_abiertos_*.csv"
    )
    all_files = sorted(glob.glob(file_pattern, recursive=True))

    if not all_files:
        raise FileNotFoundError(
            f"No se encontraron archivos RAW en {input_path}"
        )

    # Conservar los valores originales y añadir su archivo de procedencia.
    df_list = []

    for archivo in all_files:
        df_mes = pd.read_csv(archivo, encoding="latin1", dtype=str)
        df_mes["archivo_origen"] = Path(archivo).name
        df_list.append(df_mes)

    return pd.concat(df_list, ignore_index=True)

def register_dataset_version(
    df: pd.DataFrame,
    name: str,
    source: str,
    context: str,
) -> None:
    """Registra una versión del dataset como entrada de MLflow."""

    # Crear la referencia al dataset y su procedencia.
    dataset = mlflow.data.from_pandas(
        df,
        source=source,
        name=name,
    )

    # Registrar el dataset en la ejecución activa de MLflow.
    mlflow.log_input(dataset, context=context)


@app.command()
def main(

    # --- REPLACE DEFAULT PATHS AS APPROPRIATE ---
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = "data/interim/mibici_cleaned.parquet",
    # --------------------------------------------
):
    logger.info("Iniciando el procesamiento del dataset...")
    
    # 1. Cargar los archivos originales de MiBici.
    logger.info("Cargando archivos CSV desde raw...")

    # Registrar la procedencia de los archivos originales.
    # Se conserva la ruta para identificar la fuente RAW.
    raw_source = str(input_path)
    # Leer una representación de los CSV originales para MLflow.
    df_raw = load_raw_snapshot(input_path)

    try:
        df = load_raw_data(input_path)
    except FileNotFoundError as error:
        logger.error(str(error))
        return

    logger.info(f"Datos cargados: {df.shape[0]} filas.")

    # 2. Normalización de columnas
    logger.info("Normalizando encabezados...")
    df = normalize_columns(df)

    # 3. Limpieza de viajes
    logger.info("Aplicando reglas de limpieza...")
    df = clean_trips(df)
    logger.info(f"Datos después de limpieza: {df.shape[0]} filas.")

    # 4. Guardar archivo
    logger.info(f"Guardando dataset limpio en {output_path}...")
    # Asegurar que el directorio exista
    os.makedirs(output_path.parent, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    # 5. Registro de versiones en MLflow.
    logger.info("Registrando en MLflow...")
    mlflow.set_experiment("MiBici_Data_Preparation")

    with mlflow.start_run(run_name="data_cleaning_v1"):
        mlflow.log_param("raw_data_path", str(input_path))
        mlflow.log_artifact(str(output_path), artifact_path="datasets")

        # Registrar los datos originales antes de la limpieza.
        register_dataset_version(
            df=df_raw,
            name="mibici_raw",
            source=raw_source,
            context="raw",
        )

        # Registrar la versión limpia como entrada de MLflow.
        register_dataset_version(
            df=df,
            name="mibici_cleaned",
            source=str(output_path),
            context="cleaned",
        )
        
    logger.success("Procesamiento del dataset completado.")

if __name__ == "__main__":
    app()