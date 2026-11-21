
import pandas as pd

from mibici.dataset import parse_trip_dates


def test_parse_trip_dates_septiembre_2024():
    df = pd.DataFrame({
        "Inicio_del_viaje": ["05/09/2024 14:30"],
        "Fin_del_viaje": ["05/09/2024 15:00"],
    })

    resultado = parse_trip_dates(df, "datos_abiertos_2024_09.csv")

    assert resultado["Inicio_del_viaje"].iloc[0] == pd.Timestamp(
        "2024-09-05 14:30:00"
    )
    assert resultado["Fin_del_viaje"].iloc[0] == pd.Timestamp(
        "2024-09-05 15:00:00"
    )


from mibici.dataset import normalize_columns


def test_normalize_columns():
    df = pd.DataFrame({
        "Viaje_Id": [1, 2, 3],
        "Año_de_nacimiento": [1990, None, None],
        "AÃ±o_de_nacimiento": [None, 1985, None],
        "A}äe_nacimiento": [None, None, 2000],
    })

    resultado = normalize_columns(df)

    assert "anio_nacimiento" in resultado.columns
    assert resultado["anio_nacimiento"].tolist() == [
        1990, 1985, 2000
    ]
    assert "viaje_id" in resultado.columns
    assert "Año_de_nacimiento" not in resultado.columns


from mibici.dataset import clean_trips


def test_clean_trips():
    df = pd.DataFrame({
        "viaje_id": [1, 2, 3],
        "inicio_del_viaje": pd.to_datetime([
            "2024-09-05 10:00",
            "2024-09-05 11:00",
            "2024-09-05 12:00",
        ]),
        "fin_del_viaje": pd.to_datetime([
            "2024-09-05 10:30",
            "2024-09-05 10:45",
            "2024-09-05 12:20",
        ]),
        "genero": ["M", "F", None],
        "anio_nacimiento": [1990, 1985, None],
    })

    resultado = clean_trips(df)

    assert resultado["viaje_id"].tolist() == [1, 3]
    assert resultado.loc[
        resultado["viaje_id"] == 3, "anio_nacimiento"
    ].isna().all()
    assert resultado.loc[
        resultado["viaje_id"] == 3, "genero"
    ].iloc[0] == "Desconocido"


from mibici.dataset import load_raw_data


def test_load_raw_data(tmp_path):
    """Verifica la lectura y combinación de archivos CSV."""

    # Crear una carpeta temporal para simular los datos originales.
    carpeta = tmp_path / "2024"
    carpeta.mkdir()

    # Preparar dos archivos con fechas en formatos diferentes.
    archivos = {
        "datos_abiertos_2024_09.csv": "05/09/2024 14:30",
        "datos_abiertos_2024_11.csv": "2024-11-05 14:30:00",
    }

    # Guardar los archivos de prueba sin utilizar datos reales.
    for nombre, fecha in archivos.items():
        df = pd.DataFrame({
            "Viaje_Id": [1],
            "Inicio_del_viaje": [fecha],
            "Fin_del_viaje": [fecha],
        })
        df.to_csv(carpeta / nombre, index=False, encoding="latin1")

    # Ejecutar la función de carga.
    resultado = load_raw_data(tmp_path)

    # Verificar que se conservaron ambos registros.
    assert len(resultado) == 2

    # Verificar que las fechas se interpretaron correctamente.
    assert resultado["Inicio_del_viaje"].notna().all()
    assert resultado["Fin_del_viaje"].notna().all()


def test_load_raw_snapshot(tmp_path):
    """Verifica que los datos RAW conservan sus valores originales."""
    from mibici.dataset import load_raw_snapshot

    # Crear una carpeta temporal con un CSV de prueba.
    carpeta = tmp_path / "2024"
    carpeta.mkdir()

    df = pd.DataFrame({
        "Viaje_Id": ["123"],
        "Inicio_del_viaje": ["05/09/2024 14:30"],
        "Año_de_nacimiento": ["1990"],
    })

    archivo = carpeta / "datos_abiertos_2024_09.csv"
    df.to_csv(archivo, index=False, encoding="latin1")

    # Leer el archivo sin aplicar transformaciones.
    resultado = load_raw_snapshot(tmp_path)

    # Comprobar que los valores y encabezados se conservan.
    assert len(resultado) == 1
    assert resultado.loc[0, "Inicio_del_viaje"] == "05/09/2024 14:30"
    assert resultado.loc[0, "Viaje_Id"] == "123"
    assert "Año_de_nacimiento" in resultado.columns
    assert resultado.loc[0, "archivo_origen"] == "datos_abiertos_2024_09.csv"
