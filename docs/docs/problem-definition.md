# Definición del problema

## Contexto

MiBici es el sistema de bicicletas públicas del Área Metropolitana de Guadalajara.
El conjunto de datos abiertos del servicio contiene información histórica de viajes realizados desde
diciembre de 2014 y está organizado en archivos CSV por mes y año.

Los registros permiten analizar patrones de movilidad asociados con estaciones de origen y destino, horarios de uso y comportamiento temporal de la demanda. Debido
al volumen y continuidad de los datos, el dataset resulta adecuado para desarrollar un proyecto de Machine Learning con enfoque MLOPs, donde no solo se busca construir un modelo predictivo,
sino también garantizar trazabilidad, reproducibilidad y una evolución controlada del pipeline de datos y modelos. 

En este proyecto se utilizará la información historica de MiBici para analizar el comportamiento de la demanda y construir una solución que permita estimar el número de viajes iniciados
en una estación durante un periodo de tiempo determinado.

## Problema de negocio

La demanda de bicicletas en MiBici no es constante, ya que cambia dependiendo de la 
estación, la hora del día, el día de la semana y otros patrones de movilidad. Esto puede 
hacer que algunas estaciones tengan mucha demanda en ciertos horarios, mientras que otras 
tengan menos movimiento.

Cuando la disponibilidad de bicicletas no se encuentra alineada con los patrones de uso del sistema, pueden presentarse estaciones con pocas unidades disponibles o estaciones donde se acumulen bicicletas. Esto puede afectar la experiencia de los usuarios y generar una mayor necesidad de redistribución entre estaciones.

Por esta razón, contar con una estimación anticipada de la demanda puede ayudar a tener una 
mejor idea de qué estaciones podrían necesitar más atención en determinados horarios. La 
idea del proyecto no es automatizar directamente la redistribución de bicicletas, sino 
generar una predicción que pueda servir como apoyo para la planeación y la toma de decisiones.

## Problema de Machine Learning

El problema se puede plantear como una tarea de aprendizaje supervisado de tipo regresión, ya que buscamos estimar una cantidad numérica: el número de viajes que se espera que inicien en una estación durante un periodo de tiempo determinado. 

Para construir esta predicción se pueden utilizar variables relacionadas con la estación y con el comportamiento temporal de la demanda, por ejemplo la hora del día, el día de la semana, el mes y otros patrones obtenidos a partir del histórico de viajes. 

La unidad de análisis propuesta será una combinación de estación y hora. Cada registro del dataset preparado representará el número de viajes iniciados en una estación durante una hora determinada. Esto significa que , en lugar de trabajar con un registro por viaje individual, los datos se transformarán para representar cuántos viajes iniciaron en cada estación durante cada intervalo definido.

La variable objetivo será el número de viajes iniciados en una estación durante una hora determinada, es decir, la demanda observada a partir de los viajes registrados históricamente. Con eta estructura podremos entrenar un modelo base y después comparar distintas estrategias en las siguientes fases del proyecto. 


## Propuesta de valor

La propuesta de valor del proyecto consiste en utilizar el histórico de viajes de MiBici para generar estimaciones de demanda por estación y periodo de tiempo. Estas estaciones peuden servir como apoyo para identidicar de forma anticipada qué estaciones podrían tener mayor movimiento y, con ello, aportar información útil para la planeación operativa. 

El valor principal no está solamente en generar una predicción, sino en construir un proceso reproducible que permita transformar los datos, entrenar modelos, registrar resultados y evolucionar la salución de manera controlada a lo largo del tiempo. 

De esta forma, el proyecto busca combinar una necesidad operativa real con buenas prácticas de MLOps, de manera que la solución pueda mantenerse, compararse y mejorarse conforme se incorporen nuevos datos o nuevas versiones del modelo. 

## Alcance inicial

Para esta primera fase se trabajará con una parte representativa del histórico de MiBici, sin necesidad de utilizar todos los registros disponibles desde 2014. EL objetivo es contar con suficiente informacióm para identificar patrones recientes de demanda y construir un pipeline reproducible que después pueda ampliarse conforme avance el proyecto.

Inicialmente se propone trabajar con un periodo aproximado de 12 meses, utilizando los datos más recientes disponibles. Este rango puede ajustarse dependiendo del volumen final de los archivos y de los resultados del análisis exploratorio.

El análisis se enfocará en los viajes iniciados por estación y por periodo de tiempo. A partir de los registros originales se generará un conjunto de datos agregado que permita representar la demanda por estación y utilizarla porteriormente para entrenar un modelo base. 

En esta etapa no se busca optimizar al máximo el desempeño del modelo ni automatizar la redistribución de bicicletas. El objetivo pricnipal es dejar preparada una primera versión del pipeline de datos, un modelo baseline y los mecanismos necesarios para asegurar trazabilidad y reproducibilidad mediante GitHub y MLflow. 

## Supuestos y limitaciones

El proyecto parte del supuesto de que el histórico de viajes de MiBici contiene suficiente información para identificar patrones temporales y de uso por estación que permitan construir una primera estimación de demanda. 

En esta fase se trabajará principalmente con la información disponible dentro del propio dataset. No se incorporarán inicialmente variables externas como clima, tráfico, eventos especiales o disponibilidad en tiempo real de bicicletas, aunque podrían considerrse en etapas posteriores si se identifica que aportan valor al modelo. 

También se asume que los datos históricos son representativos del comportamiento reciente del sistema, aunque pueden existir cambios en estaciones, infraestructura, hábitos de movilidad o condiciones operativas que afecten esa relación a lo largo del tiempo. 

Otra limitación importante es que el modelo estimará, pero no resolverá directamente el problema de redistribución de bicicletas. La predicción funcionará como una entrada para apoyar decisiones operativas, no como una optimización automática de movimientos entre estaciones. 

Finalmente, el desempeño del modelo dependerá de la calidad de los datos, de la granualiridad temporal seleccionada y de las variables utilizadas. Por esta razón, durante las siguientes fases será necesario evaluar distintas versiones del dataset, comparar modelos y monitorear posibles cambios en la distribución de los datos. 