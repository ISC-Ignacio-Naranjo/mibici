
# Versionado de datasets MiBici — Fase 2

## Objetivo

Documentar la trazabilidad de los datos de MiBici mediante MLflow, desde los archivos originales (RAW) hasta los datos preparados para entrenamiento (PROCESSED).

## Transformaciones entre versiones

| Versión | Transformaciones | Resultado | Estado |
|---|---|---|---|
| RAW | Lectura de archivos CSV sin limpieza; se conserva el nombre del archivo de procedencia. | Representación de los registros originales. | Verificado con muestra |
| CLEANED | Conversión de fechas según el mes, unificación de encabezados, sustitución de género faltante por "Desconocido" y eliminación de viajes con fechas inválidas o duración negativa. | Dataset limpio en formato Parquet. | Verificado con muestra |
| PROCESSED | Agregación de demanda por estación y hora, incorporación de horas con demanda cero, variables temporales y rezagos de 1 y 24 horas. | Dataset de características para entrenamiento. | Pendiente de validación integrada |

## Registro en MLflow

Experimento: `MiBici_Data_Preparation`

Ejecución: `data_cleaning_v1`

Se utiliza `mlflow.log_input()` para registrar los datasets y sus metadatos.

### Validación funcional

La prueba se realizó con 1,000 viajes de septiembre de 2024.

| Dataset | Registros | Columnas | Identificador MLflow |
|---|---:|---:|---|
| mibici_raw | 1,000 | Pendiente de comprobar | 5d76ae46 |
| mibici_cleaned | 1,000 | 8 | c913493d |

Las versiones RAW y CLEANED aparecen registradas en una misma ejecución de MLflow.

La validación del dataset completo de 2024–2025 continúa pendiente.

## Pruebas automatizadas

Se ejecutó `python -m pytest tests/test_data.py -v`.

Resultado: **5 pruebas superadas**.

Las pruebas cubren interpretación de fechas, normalización de columnas, reglas de limpieza, carga de archivos y lectura de datos RAW.

## Evidencias

- `evidencia_1_data_cleaning_v_1.jpg`: metadatos y esquema del dataset CLEANED.
- `evidencia_2_versionado_raw_cleaned.jpg`: registro conjunto de RAW y CLEANED en MLflow.

## Trabajo pendiente

- Validar el pipeline con el conjunto completo de datos.
- Verificar el registro de PROCESSED realizado en el módulo de ingeniería de características.
- Integrar la trazabilidad de las tres versiones en la documentación final del equipo.

## Nota metodológica

El registro RAW representa los valores leídos de los CSV originales, pero no constituye una copia binaria de los archivos fuente. El registro mediante MLflow documenta procedencia, esquema e identificadores de los datos; no sustituye un sistema de almacenamiento y versionado de archivos originales.
