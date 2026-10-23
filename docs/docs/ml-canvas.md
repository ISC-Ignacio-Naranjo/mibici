# ML Canvas

## 1. Proposición de valor

El proyecto busca utilizar el histórico de viajes de MiBici para estimar la demanda observada
por estación y por hora, de manera que sea posible anticipar qué estaciones podrían presentar
mayor nivel de actividad en determinados periodos.

Esta información puede servir como apoyo para los responsables de operación y planeación del
sistema, al proporcionar una referencia adicional para identificar estaciones que podrían
requerir mayor atención dentro de los procesos de disponibilidad y redistribución de bicicletas.

El valor del proyecto no se limita únicamente a generar una predicción. También se busca
construir una solución reproducible y trazable, donde los datos, transformaciones, experimentos
y modelos puedan evolucionar de forma controlada mediante buenas prácticas de MLOps.

Los usuarios objetivo de la solución serían las áreas responsables de operación y planeación
de MiBici, mientras que los usuarios del sistema de bicicletas serían beneficiarios indirectos
en un posible escenario de adopción.

## 2. Fuentes de datos

La principal fuente de información será el conjunto de datos abierto de MiBici, el cual contiene
registros históricos de viajes realizados dentro del sistema de bicicletas públicas del Área
Metropolitana de Guadalajara.

Los datos están disponibles en archivos CSV organizados por periodos e incluyen información
asociada a los viajes realizados, como su origen, destino y componentes temporales. La
estructura exacta de los archivos será validada durante el perfilado inicial para identificar
tipos de datos, valores faltantes, duplicados, inconsistencias y posibles cambios de esquema
entre periodos.

Para la primera fase se propone trabajar inicialmente con aproximadamente 12 meses de
información reciente, lo que permite cubrir un ciclo anual completo sin incorporar desde el
inicio todo el histórico disponible. El periodo definitivo se podrá ajustar después del
perfilado inicial del dataset, dependiendo de la calidad de los datos, posibles cambios de
estructura entre archivos y el volumen de información disponible.

En esta primera versión no se incorporarán fuentes externas como clima, tráfico o eventos
especiales, aunque podrán evaluarse posteriormente si se identifica que aportan valor al modelo.

## 3. Recolección de datos

Los datos se obtendrán a partir de los archivos publicados por MiBici dentro de su portal de
datos abiertos.

Los archivos originales se conservarán sin modificaciones dentro de la capa de datos crudos
(`data/raw`). A partir de ellos se realizarán procesos de validación, limpieza y transformación
para generar versiones intermedias y posteriormente un dataset listo para modelado.

Debido a que MiBici continúa publicando información de nuevos periodos, el proyecto podrá
incorporar datos más recientes en futuras etapas. Esto permitirá actualizar el dataset,
evaluar el comportamiento del modelo con información nueva y analizar posibles cambios en
los patrones de movilidad.

La organización de los datos seguirá las capas definidas en la estructura del proyecto:

- `data/raw`: archivos originales sin modificar.
- `data/interim`: datos después de procesos de limpieza y validación.
- `data/processed`: datos preparados para análisis y modelado.

Conceptualmente, esta separación es similar al flujo Bronze, Silver y Gold utilizado en una
arquitectura medallón, aunque el proyecto conservará la nomenclatura definida por Cookiecutter
Data Science.

## 4. Características

Las variables de entrada se obtendrán principalmente a partir de la estación de origen y de
la información temporal asociada a los viajes.

Después de agrupar los registros por estación y hora, se podrán generar características como:

- Estación de origen.
- Hora del día.
- Día de la semana.
- Mes.
- Indicador de fin de semana.
- Variables temporales adicionales identificadas durante el análisis exploratorio.
- Información histórica de demanda de periodos anteriores, cuando sea pertinente.

Cualquier característica basada en demanda histórica deberá construirse utilizando únicamente
información disponible antes del momento que se desea predecir. Esto es importante para evitar
data leakage y mantener un comportamiento similar al que tendría el modelo en producción.

La selección definitiva de variables se realizará después del perfilado y análisis exploratorio
del dataset.

## 5. Construcción de modelos

Durante la primera fase se construirá un modelo baseline sencillo que permita validar el flujo
completo desde la preparación de los datos hasta la generación y registro de una predicción.

El objetivo inicial no será encontrar el modelo con mejor desempeño posible, sino contar con
una referencia que permita comprobar que el pipeline funciona correctamente y que los
experimentos pueden reproducirse.

El baseline también servirá como punto de referencia mínimo para determinar si modelos
posteriores realmente aportan una mejora medible y no únicamente una mayor complejidad.

En las siguientes fases se podrán evaluar diferentes algoritmos, realizar ajustes de
hiperparámetros y comparar distintas versiones del modelo utilizando MLflow.

La elección de un modelo final no se realizará únicamente con base en una métrica. También se
considerarán aspectos como estabilidad del desempeño entre estaciones, simplicidad,
mantenibilidad y facilidad de integración dentro del pipeline de MLOps.

## 6. Tareas de Machine Learning

El problema se plantea como una tarea de aprendizaje supervisado de tipo regresión.

La variable objetivo será el número de viajes iniciados en una estación de MiBici durante una
hora determinada.

Para construir esta variable será necesario transformar los registros individuales de viajes y
agruparlos por estación de origen y periodo horario. Además, durante la preparación del dataset
se deberán representar los periodos sin viajes para que las combinaciones válidas de estación
y hora con ausencia de viajes tengan una demanda observada igual a cero.

La construcción de estos registros deberá considerar únicamente periodos en los que la estación
se encuentre activa, evitando interpretar como demanda cero aquellos periodos en los que una
estación todavía no existía o no se encontraba operando.

La unidad de análisis será, por lo tanto, una combinación de estación y hora. Cada registro del
dataset preparado representará la demanda observada para una estación durante una hora
específica.

El modelo buscará aprender patrones históricos que permitan estimar esta cantidad para periodos
posteriores.

## 7. Decisiones

La predicción generada por el modelo funcionará como una fuente adicional de información para
las personas responsables de operación y planeación del sistema MiBici.

Las estimaciones podrán utilizarse para identificar estaciones y periodos con mayor nivel de
actividad esperado y apoyar la priorización de revisiones o acciones operativas.

El modelo no decidirá automáticamente cuántas bicicletas deben moverse ni ejecutará acciones
de redistribución por sí mismo. La decisión final deberá considerar la predicción junto con
otros factores operativos que no necesariamente estarán disponibles dentro del dataset.

De esta manera, la solución se plantea como un sistema de apoyo a la toma de decisiones y no
como un sistema autónomo.

## 8. Hacer predicciones

La salida esperada será una estimación del número de viajes que iniciarán en una estación
durante una hora determinada.

Como horizonte inicial de predicción se propone estimar la demanda observada de la siguiente
hora para cada estación. La predicción se generará utilizando únicamente información disponible
antes del inicio del periodo que se desea predecir.

Por ejemplo, para una estación y una hora específica, el modelo podría estimar que se iniciarán
20 viajes durante ese periodo.

En etapas posteriores podrá evaluarse la extensión del horizonte para generar predicciones de
varias horas, dependiendo del desempeño obtenido y de las necesidades operativas identificadas.

La predicción no representa directamente el número de bicicletas disponibles ni la cantidad
exacta de bicicletas que se deben redistribuir. Representa una estimación del nivel de actividad
esperado a partir de los patrones históricos observados.

En etapas posteriores, esta capacidad podrá exponerse mediante una API e integrarse con otros
procesos o sistemas.

## 9. Evaluación offline

Para evaluar el desempeño del modelo se utilizarán métricas apropiadas para problemas de
regresión.

Como métricas principales se consideran:

- MAE (Mean Absolute Error), para medir el error promedio absoluto entre la demanda observada
  y la demanda estimada.
- RMSE (Root Mean Squared Error), para penalizar con mayor fuerza errores grandes y detectar
  predicciones con desviaciones importantes.

Debido a que el problema tiene un componente temporal, la separación de los datos para
entrenamiento y evaluación deberá respetar el orden cronológico. Los periodos más antiguos se
utilizarán para entrenamiento y los periodos posteriores para validación o prueba, evitando
que información futura esté disponible durante el entrenamiento.

Estas métricas permitirán comparar diferentes versiones del modelo. Sin embargo, la evaluación
no se limitará únicamente al resultado promedio.

También será importante revisar el desempeño entre distintas estaciones, horarios y periodos,
ya que un buen resultado global podría ocultar errores importantes en determinados segmentos
de los datos.

La selección de una versión del modelo deberá considerar tanto el desempeño técnico como la
utilidad que las predicciones puedan tener dentro del contexto operativo.

## 10. Monitorización y evaluación en vivo

Una vez que el modelo avance hacia un escenario de despliegue, su desempeño deberá evaluarse
conforme se incorporen nuevos datos de MiBici.

El dataset continúa actualizándose, por lo que será posible utilizar información de periodos
posteriores para comparar las predicciones contra la demanda observada y detectar cambios en
los patrones de uso.

Esto permitirá analizar posibles situaciones de data drift y determinar si la distribución de
los datos actuales comienza a alejarse de aquella utilizada durante el entrenamiento.

También será necesario monitorear la evolución de las métricas del modelo. Si el desempeño se
degrada de manera significativa, podrá evaluarse la necesidad de actualizar el dataset,
reentrenar el modelo o reemplazarlo por una nueva versión.

MLflow permitirá asociar a cada experimento la información correspondiente al dataset utilizado,
junto con parámetros, métricas, modelos y artefactos. En conjunto con Git y la estrategia de
manejo de datos del proyecto, esto permitirá mantener trazabilidad entre las distintas
ejecuciones del pipeline.

## Riesgos y limitaciones

Uno de los principales riesgos del proyecto es que los datos históricos representan únicamente
los viajes que sí se realizaron. Esto significa que no necesariamente reflejan toda la demanda
potencial, por ejemplo cuando un usuario no pudo iniciar un viaje porque no había bicicletas
disponibles en una estación.

También pueden existir cambios en la red de estaciones, en la infraestructura, en los hábitos
de movilidad o en las condiciones operativas que hagan que los patrones históricos cambien
con el tiempo.

Otra limitación es que en esta primera versión no se incorporarán variables externas como
clima, tráfico o eventos especiales, aunque podrían influir en la demanda.

Desde el punto de vista técnico, existe el riesgo de introducir data leakage si se utiliza
información futura para construir las variables de entrada. Por esta razón, cualquier
característica histórica deberá generarse únicamente con información disponible antes del
momento que se desea predecir.

Finalmente, el modelo funcionará como una herramienta de apoyo y no como un sistema autónomo
de decisión. Sus resultados deberán interpretarse junto con otros criterios operativos antes
de tomar acciones sobre la distribución de bicicletas.