
# Roles, actividades e interacciones

## 1. Organización del trabajo

El desarrollo del proyecto MiBici requiere la participación de diferentes roles relacionados con el cíclo de vida de una solución de Machine Learning. Para este proyecto se consideran los roles de Data Engineer, Data Scientist, Machine Learning Engineer, Software Engineer y Site Reliability Engineer (SRE). 

Estos roles representan responsabilidades dentro del proyecto y no necesariamente una asignación exclusiva de una persona por rol. Debido al tamaño del equipo, un integrante puede participar en actividades correspondientes a más de un rol y colaborar con otros miembros durante diferentes etapas del desarrollo. 

El objetivo de esta organización es identificar claramente las responsabilidades necesarias para llevar el modelo desde los datos originales hasta una solución reproducible, evaluable y preparada para una posible integración o despliegue. 

## 2. Data Engineer

El Data Engineer es responsable de preparar y mantener el flujo de datos utilizado por el proyecto.

Dentro de MiBici, sus principales actividades incluyen:

- Obtener los archivos históricos de viajes publicados por MiBici.
- Conservar los archivos originales sin modificaciones dentro de `data/raw`.
- Analizar el esquema, tipos de datos, valores faltantes, duplicados e inconsistencias. 
- Identificar posibles cambios de estructura entre diferentes periodos. 
- Implementar procesos de limpieza y transformación reproducibles. 
- Generar los datasets intermedios almacenados en `data/interim`.
- Preparar la información necesaria para construir el dataset de modelado. 
- Validar que las transformaciones mantengan la calidad y consistencia de los datos. 
- Documentar las decisiones realizadas durante el proceso de preparación. 

Este rol también deberá considerar que la unidad final de análisis será estación por hora, incluyendo periodos válidos con cero viajes cuando corresponda y evitando generar registros para periodos en los que una estación todavía no se encontraba activa.

## 3. Data Scientist

El Data Scientist es responsable de analizar los datos y desarrollar la lógica necesaria para convertirlos en un problema de Machine Learning. 

Dentro del proyecto sus principales actividades incluyen:

- Realizar el análisis exploratorio de datos. 
- Analizar los patrones temporales de uso de las estaciones. 
- Evaluar la distribución de la variable objetivo. 
- Identificar variables que puedan aportar información al modelo. 
- Diseñar caracteristicas derivadas de la información temporal.
- Evaluar posibles características históricas de demanda sin introducir data leakage. 
- Definir la estrategia de separación temporal entre entrenamiento, validación y prueba. 
- Seleccionar métricas adecuadas para evaluar los modelos. 
- Analizar los resultados obtenidos y no limitar la evaluación únicamente a una métrica global. 
- Comparar el desempeño entre diferentes estaciones, horarios o periodos cuando sea necesario. 

El Data Scientist utilizará MAE y RMSE como métricas iniciales de evaluación y trabajará con una estrategia de validación que respete el orden cronológico de los datos. 

## 4. Machine Learning Engineer

El Machine Learning Engineer es responsable de convertir el trabajo de experimentación en un flujo reproducible de entrenamiento y evaluación. 

Dentro del proyecto sus principales actividades incluyen:

- Implementar un modelo baseline para establecer un punto de referencia. 
- Integrar el dataset proesado con el proceso de entrenamiento.
- Parametrizar los experimentos para evitar configuraciones escritas directamente en el código. 
- Registrar experimentos mediante MLflow.
- Asociar parámetros, métricas, información del dataset y artefactos con cada ejecución. 
- Comparar diferentes versiones de los modelos. 
- Mantener reproducibilidad entre ejecuciones. 
- Preparar la evolución del pipeline hacia MLflow Projects en fases posteriores. 
- Participar en el registro y administración de versiones de modelos conforme avance el proyecto. 

EL objetivo inicial de este rol no será encontrar inmediatamente el mejor modelo posible, sino establecer un proceso de entrenamiento confiable sobre el cual puedan realizarse mejoras de forma controlada. 

## 5. Software Engineer

El Software Engineer es responsable de mantener la calidad y estructura del código que soporta la solución. 

Dentro del proyecto sus principales actividades incluyen:

- Mantener una estructura modular basada en Cookiecuter Data Science.
- Convertir progresivamente el código exploratorio de notebooks en funciones, módulos y scripts reutilizables. 
- Evitar duplicación de lógica y configuraciones escritas directamente en el código.
- Aplicar convenciones de nombres y buenas prácticas de desarrollo.
- Utilizar herramientas de formato y análisis estático como Black, isort y Flake9. 
- Mantener las dependencias y configuración del entorno de ejecución. 
- Implementar pruebas conforme los componentes sean refactorizados. 
- Preprarar los componentes necesararios para que el modelo pueda integrarse posteriormente con una API y otros sistemas. 
- Mantener el código preparado para su ejecución fuera de un notebook.

Este rol ayuda a evitar que el proyecto permanezca únicamente como una prueba de concepto y facilita su evolución hacia una solución mantenible. 

## 6. Site Reliability Engineer (SRE)

El SRE se enfoca en la confiabilidad, observabilidad y operación futura de la solución. 

Aunque varias de sus responsabilidades tendrán mayor importancia durante las siguientes fases, desde la primera etapa se consideran actividades como:

- Definir aspectos que deberán ser observables durante la ejecución del pipeline. 
- Identificar información necesaria para diagnosticar errores en futuras ejecuciones. 
- Participar en la definición de mecanismos de monitoreo. 
- Considerar la detección de cambios en la distribución de los datos. 
- Evaluar posteriormente posibles situaciones de data drift.
- Supervisar la evolución de las métricas del modelo una vez que exista una versión desplegada.
- Definir criterios que permitan determinar cuándo investigar, reentrenar o reemplazar un modelo. 
- Colaborar en la preparación de una solución portable mediante contenedores en etapas posteriores. 

En este proyecto, la disponibilidad continua de nuevos datos de MiBici permitirá utilizar periodos futuros para evaluar cambios reales en los patrones de movilidad y analizar posibles situaciones de degradación del modelo. 

## 7. Interacción entre roles

Los roles definidos no trabajan de manera aislada. El proyecto requiere un flujo continuo de información y retroalimentación entre ellos. 

El Data Engineer prepara y valida los datos que posteriormente serán utilizados por el Data Scientist. Si durante el análisis exploratorio se detectan problemas de caldiad, inconsistencias o necesidades adicionales de transformación es utilizada por el Machine Learning Engineer para construir un pipeline reproducible de entrenamiento, registrar experimentos y comparar resultados. 

El Machine Learning Engineer trabaja en conjunto con el Software Engineer para transformar la Experimentación en componentes reutilizables, parametrizados y mantenibles. El Software Engineer también asegura que la estructura del código permita incorporar pruebas, automatización y futuras interfaces de integración. 

EL SRE utiliza información generada por el pipeline, los experimentos y el modelo para definir qué elementos deberán monitorearse cuando la solución avance hacia un escenario de despliegue. A su vez, los cambios detectados en producción pueden generar nuevas necesidades de datos, reentrenamiento o ajustes al modelo, cerrando el ciclo con los demas roles. 

De manera simplidicada, la interacción principal puede representarse como: 

`Datos → preparación → análisis → entrenamiento → integración → monitoreo → retroalimentación`

Este flujo no es estrictamente lineal. Los resultados obtenidos en una erapa pueden generar cmabios en etapas anteriores, por lo que el proyecto se desarrollará de forma iterativa. 

## 8. Matriz de actividades e interacción durante la Fase 1

Las responsabilidades de los roles se traducen durante la primera fase en actividades y
entregables concretos. Debido a que el equipo está formado por cuatro integrantes, una misma
persona puede participar en más de un rol dependiendo de la actividad realizada.

| Actividad | Rol principal | Entregable o evidencia | Interacción principal |
| --- | --- | --- | --- |
| Definición del problema y propuesta de valor | Data Scientist | Definición del problema y ML Canvas | Software Engineer, ML Engineer |
| Obtención y organización de los datos | Data Engineer | Datos originales organizados en `data/raw` | Data Scientist |
| Perfilado y validación inicial de datos | Data Engineer | Análisis de esquema, tipos, valores faltantes, duplicados e inconsistencias | Data Scientist |
| Limpieza y transformación | Data Engineer | Dataset intermedio en `data/interim` y procesos reproducibles | Data Scientist |
| Análisis exploratorio de datos | Data Scientist | Notebook o reporte de EDA | Data Engineer, ML Engineer |
| Ingeniería y selección inicial de características | Data Scientist | Dataset y variables propuestas para modelado | Data Engineer, ML Engineer |
| Construcción del dataset estación-hora | Data Engineer / Data Scientist | Dataset preparado en `data/processed` | ML Engineer |
| Definición de evaluación temporal y métricas | Data Scientist | Estrategia de evaluación, MAE y RMSE | ML Engineer |
| Implementación del modelo baseline | ML Engineer | Primera versión entrenable del modelo | Data Scientist |
| Registro de experimentos | ML Engineer | Ejecuciones y métricas registradas en MLflow | Data Scientist, Software Engineer |
| Organización y modularización del código | Software Engineer | Estructura Cookiecutter, módulos y scripts reutilizables | Data Engineer, ML Engineer |
| Gestión del entorno y dependencias | Software Engineer | `pyproject.toml` y entorno reproducible | Todos los roles técnicos |
| Control de versiones e integración | Software Engineer | Branches, commits y Pull Requests en GitHub | Todos los roles |
| Definición inicial de aspectos observables | SRE | Criterios de monitoreo, métricas y futuros indicadores de drift | ML Engineer, Data Scientist |
| Identificación de criterios de actualización del modelo | SRE / ML Engineer | Estrategia inicial de monitoreo y reentrenamiento | Data Scientist |


## 9. Colaboración durante la Fase 1

Durante la primera fase, las actividades se concentran principalmente en establecer las bases del proyecto:

- Definición del problema y propuesta de valor;
- Preparación inicial de datos;
- Análisis exploratorio y preprocessing;
- Construcción del dataset de modelado;
- Implementación de un baseline;
- Registro inicial de experimentos con MLflow;
- Organización del repositorio y del entorno reproducible;
- Documentación de decisiones, resultados y limitaciones.

Las actividades se desarrollarán mediante ramas independientes en Git y posteriormente se integrarán mediante Pull Requests. Este flujo permitirá revisar los cambios realizados por los integrantes, mantener trazabilidad de las contribuciones y reducir conflictos durante la integración. 

La participación de cada integrante podrá abarcar diferentes responsabilidades de los roles descritos anteriormente. Por esta razón, la asignación de trabajo del equipo se base en actividades y enregables concretos, mientras que los roles sirven para identificar cómo esas actividades forman parte del ciclo de vida completo de Machine Learning.

Las contribuciones realizadas por los integrantes quedarán registradas mediante commits,
branches y Pull Requests dentro del repositorio del proyecto. Esta evidencia permitirá
relacionar las actividades descritas con el trabajo realizado durante la Fase 1, sin asumir
que cada integrante desempeña exclusivamente un único rol.
