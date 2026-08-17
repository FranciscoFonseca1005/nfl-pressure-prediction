# Predicción de presión del pass rush de la NFL al momento del snap

## Informe técnico

**Autor:** Francisco Miguel Fonseca Martínez  
**Fecha de evaluación:** 17 de agosto de 2026  
**Repositorio:** [nfl-pressure-prediction](https://github.com/FranciscoFonseca1005/nfl-pressure-prediction)  
**Commit de la evaluación final:** `57b088e41425b18c844e18cd34ba6cb73f8dae7e`

> **Estado del proyecto:** La evaluación final fuera de tiempo se completó sobre la semana 8 de la NFL sin realizar cambios posteriores al test en el modelo, las variables, los hiperparámetros, la calibración o el umbral de clasificación.

---

## Resumen ejecutivo

Este proyecto desarrolla y valida una solución de Machine Learning que estima, al momento del snap, la probabilidad de que un pass rusher individual genere presión sobre el quarterback más adelante en la jugada.

En términos de futbol americano, un **pass rusher** es un jugador defensivo que intenta alcanzar o incomodar al quarterback, es decir, al jugador ofensivo que normalmente recibe el balón y realiza el pase. Una **presión** se define a partir de las etiquetas de PFF como al menos un hurry, un golpe al quarterback o un sack. El modelo produce una predicción por cada combinación de pass rusher y jugada utilizando únicamente información disponible cuando el balón entra en juego.

El análisis utilizó datos de tracking de la NFL y datos de scouting de PFF correspondientes a las semanas 1 a 8. Después de validar los datos y excluir jugadas sin un snap manual válido, la tabla analítica final contuvo:

- 36,259 observaciones de pass rushers.
- 8,531 jugadas únicas, identificadas mediante `gameId + playId`.
- 4,214 resultados positivos de presión.
- Una prevalencia general de presión de 11.622%.
- 51 predictores elegibles: 50 variables numéricas de tracking y una variable categórica de alineación.

Se utilizó un diseño temporal estricto:

| Propósito | Semanas NFL | Filas | Jugadas únicas | Positivos | Tasa de presión |
|---|---:|---:|---:|---:|---:|
| Entrenamiento inicial | 1-6 | 27,950 | 6,587 | 3,283 | 11.746% |
| Validación y selección del umbral | 7 | 3,865 | 913 | 427 | 11.048% |
| Test final no utilizado previamente | 8 | 4,444 | 1,031 | 504 | 11.341% |

Se conservó un pipeline de regresión logística sin ponderación de clases en lugar de candidatos más complejos de Extra Trees y histogram gradient boosting. Extra Trees produjo una mejora marginal de apenas 0.001790 en average precision sobre validación, y el intervalo bootstrap de 95% para esa ganancia incluyó el cero. Por lo tanto, la evidencia no justificó la complejidad adicional.

El umbral de clasificación se seleccionó exclusivamente sobre la semana 7 maximizando F1 y se bloqueó permanentemente en `0.12485514806532311`. Posteriormente, el modelo seleccionado se clonó y se ajustó una sola vez utilizando las semanas 1-7, antes de acceder a la semana 8.

El desempeño final de la semana 8 fue:

| Métrica | Resultado semana 8 |
|---|---:|
| Average precision | 0.157750 |
| ROC-AUC | 0.596890 |
| Log loss | 0.347812 |
| Precision | 0.158940 |
| Recall | 0.476190 |
| F1 | 0.238332 |
| Specificity | 0.677665 |
| Tasa real de presión | 0.113411 |
| Tasa de predicciones positivas | 0.339784 |

El average precision final fue 1.391 veces la prevalencia de la semana 8. El decil con puntuaciones más altas presentó una tasa observada de presión de 19.101%, equivalente a 1.684 veces la prevalencia del test.

Los resultados respaldan una **capacidad de ranking modesta pero temporalmente estable**. El modelo puede ayudar a priorizar pass rushers para revisión o análisis posterior, pero su baja precisión y elevado número de falsos positivos impiden considerarlo un detector autónomo de alta confianza.

---

## 1. Definición del problema

### 1.1 Pregunta analítica

El proyecto responde la siguiente pregunta:

> Utilizando únicamente información de tracking de los jugadores y el balón disponible al momento del snap, ¿puede un modelo ordenar a los pass rushers según su probabilidad de generar presión sobre el quarterback?

Se trata de un problema de clasificación probabilística binaria:

\[
P(\text{pressure}=1 \mid X_{\text{snap}})
\]

donde:

- `pressure = 1` indica un hurry, hit o sack registrado por PFF.
- `pressure = 0` indica que ninguno de esos eventos fue registrado.
- \(X_{\text{snap}}\) contiene exclusivamente variables disponibles al momento del snap.

### 1.2 Unidad de análisis

El grano analítico es un pass rusher dentro de una jugada. Su llave única es:

```text
gameId + playId + nflId
```

Esta distinción es importante porque una misma jugada puede contener varios pass rushers. Cada jugador elegible recibe una probabilidad y una clasificación individual. Una jugada debe identificarse mediante `gameId + playId`, ya que `playId` por sí solo puede repetirse en diferentes partidos.

### 1.3 Usos previstos y excluidos

La probabilidad puede utilizarse para ordenar pass rushers, priorizar observaciones para revisión de video o análisis, estudiar descriptivamente la geometría inicial de la jugada o servir como una entrada dentro de un sistema más amplio de apoyo a decisiones.

El modelo no pretende predecir sacks de manera específica, juzgar la calidad general de un jugador, sustituir el scouting de PFF, utilizar trayectorias posteriores al snap, tomar decisiones autónomas de alta confianza ni establecer relaciones causales entre la alineación y la presión.

---

## 2. Fuentes de datos y alcance

Los archivos fuente proceden del [dataset de Kaggle proporcionado](https://www.kaggle.com/datasets/dmay01/usefuldata).

| Componente | Contenido | Función dentro del proyecto |
|---|---|---|
| Archivos Parquet de tracking | Posición, velocidad, dirección, orientación, eventos y frames de jugadores y balón | Creación de predictores al momento del snap |
| Datos de scouting PFF | Rol, alineación y resultados del pass rush | Identificación de pass rushers y definición del target |
| Información de equipos | Identificadores y metadatos de equipos | Validación y referencia |

Los datos brutos de tracking contenían 8,314,178 filas y 16 columnas distribuidas en ocho archivos Parquet. La fuente de scouting PFF contenía 188,254 registros jugador-jugada.

Los nombres físicos de los archivos de tracking no correspondían perfectamente a las semanas reales de la NFL. Por esta razón, `actual_week` se derivó de la fecha del partido en lugar de inferirse del nombre del archivo. Esto evitó construir una separación temporal incorrecta.

El alcance disponible cubre 122 partidos de las semanas 1 a 8. Las conclusiones no deben generalizarse automáticamente a toda una temporada, otra temporada, partidos de postemporada o un sistema de tracking diferente.

---

## 3. Validación de datos y construcción del target

### 3.1 Validación de las fuentes

El proceso inicial verificó que los archivos esperados estuvieran presentes y no vacíos, que los Parquet de tracking tuvieran esquemas compatibles, que cada frame contuviera exactamente 23 entidades, que las secuencias de frames no presentaran huecos inesperados, que las llaves jugador-jugada de PFF fueran completas y únicas, y que los registros de tracking y PFF coincidieran uno a uno.

Los indicadores PFF se comprobaron contra dominios binarios válidos. La ausencia estructural de valores se preservó en lugar de imputarse globalmente, y los alias históricos de equipos se conservaron como metadatos de referencia. Las observaciones inusuales se documentaron mediante controles y notas, en vez de eliminarse silenciosamente.

### 3.2 Selección del snap

Las variables se construyeron en el evento manual `ball_snap`, que representa el momento en que el centro inicia la jugada entregando el balón al quarterback.

De 8,557 jugadas fuente, 8,532 tenían un marcador manual válido y 25 fueron excluidas. Las filas de Pass Rush disminuyeron de 36,362 a 36,259; solamente se retiraron 103 registros, aproximadamente 0.283%. Utilizar un evento explícito evita combinar accidentalmente información previa y posterior al snap.

### 3.3 Target de presión

El target binario se construyó como:

```text
pressure = pff_hurry OR pff_hit OR pff_sack
```

| Propiedad del target | Resultado |
|---|---:|
| Filas totales | 36,259 |
| Presiones positivas | 4,214 |
| Casos negativos | 32,045 |
| Prevalencia de presión | 11.622% |
| Valores faltantes | 0 |
| Clases observadas | 0 y 1 |

Los componentes utilizados para construir el target se excluyeron de los predictores para impedir fuga de información.

---

## 4. Tabla analítica e ingeniería de variables

La tabla analítica procesada se encuentra en `data/processed/pass_rush_analytical_table.parquet`. Contiene 36,259 filas y 70 columnas documentadas. Cada columna se asignó exactamente a un rol: identificador, metadato temporal, target, predictor o campo exclusivo de auditoría.

### 4.1 Contrato de predictores

El contrato aprobado contiene 50 predictores numéricos y un predictor categórico, `rusher_position_lined_up`, para un total de 51 variables en un orden almacenado permanentemente. No existían valores faltantes, valores no finitos ni predictores constantes.

### 4.2 Familias de variables

| Familia | Ejemplos | Interpretación |
|---|---|---|
| Geometría rusher-quarterback | Distancia, separación longitudinal y desplazamiento lateral | Posición inicial del rusher con respecto al quarterback |
| Geometría rusher-balón | Posición relativa al balón | Ubicación del defensor respecto al punto del snap |
| Profundidad del quarterback | Posición detrás del balón | Posicionamiento inicial del quarterback |
| Bloqueador más cercano | Distancia y alineación direccional | Relación entre el rusher y el probable bloqueador más cercano |
| Movimiento | Velocidad, componentes y closing speed | Dirección e intensidad de movimiento al momento del snap |
| Codificación direccional | Transformaciones seno y coseno | Representación continua de ángulos circulares |
| Alineación PFF | `rusher_position_lined_up` | Categoría de alineación específica del dataset |

Las variables numéricas se estandarizaron dentro del pipeline. La variable categórica se transformó mediante one-hot encoding. Identificadores, campos de auditoría, componentes del target, semanas y resultados se excluyeron del modelo.

---

## 5. Análisis exploratorio

### 5.1 Desbalance de clases y estabilidad temporal

Solamente 11.622% de las observaciones generaron presión. Por ello, un clasificador podría obtener accuracy elevada prediciendo casi siempre que no habrá presión. Se priorizó average precision porque evalúa qué tan bien las puntuaciones altas concentran la clase positiva poco frecuente.

![Prevalencia semanal de presión](report_weekly_pressure_rate.png)

| Semana | Filas | Jugadas únicas | Positivos | Tasa de presión |
|---:|---:|---:|---:|---:|
| 1 | 4,988 | 1,172 | 598 | 11.989% |
| 2 | 4,498 | 1,062 | 521 | 11.583% |
| 3 | 4,809 | 1,139 | 606 | 12.601% |
| 4 | 4,733 | 1,108 | 557 | 11.768% |
| 5 | 4,681 | 1,105 | 528 | 11.280% |
| 6 | 4,241 | 1,001 | 473 | 11.153% |
| 7 | 3,865 | 913 | 427 | 11.048% |
| 8 | 4,444 | 1,031 | 504 | 11.341% |

La prevalencia permaneció dentro de un intervalo relativamente estrecho, lo que reduce la preocupación por un cambio severo de la tasa objetivo entre desarrollo y test.

### 5.2 Diferencias de variables al momento del snap

| Variable | Media sin presión | Media con presión | Diferencia |
|---|---:|---:|---:|
| Distancia rusher-QB | 6.504 | 6.936 | +0.433 |
| Separación longitudinal rusher-QB | 5.076 | 5.278 | +0.203 |
| Distancia al bloqueador más cercano | 2.600 | 2.878 | +0.278 |
| Velocidad del rusher hacia el QB | 0.229 | 0.299 | +0.070 |
| Closing speed rusher-QB | 0.199 | 0.270 | +0.071 |

Las presiones mostraron algo más de movimiento hacia el quarterback y mayor velocidad de acercamiento al snap. Estas son comparaciones descriptivas marginales, no efectos causales.

---

## 6. Diseño experimental

Una separación aleatoria de filas no representaría la predicción de semanas futuras. Se utilizaron las semanas 1-6 para entrenamiento inicial, la semana 7 para validar modelos y seleccionar el umbral, y la semana 8 para un único test final. Ninguna llave `gameId + playId` apareció en más de una partición.

La calidad probabilística se evaluó mediante average precision, ROC-AUC y log loss. Brier score se utilizó descriptivamente para calibración. Después de seleccionar el umbral se evaluaron precision, recall, F1, specificity, la tasa de predicciones positivas y la matriz de confusión.

---

## 7. Modelos baseline y avanzados

La comparación baseline incluyó un clasificador dummy basado en prevalencia, regresión logística sin ponderación y regresión logística balanceada. Los predictores numéricos se estandarizaron y la alineación se codificó mediante one-hot encoding dentro de los pipelines.

| Modelo | Average precision | ROC-AUC | Log loss | Seleccionado |
|---|---:|---:|---:|---|
| Dummy prior | 0.110479 | 0.500000 | 0.347754 | No |
| Regresión logística sin ponderación | 0.151899 | 0.600006 | 0.342382 | Sí |
| Regresión logística balanceada | 0.151749 | 0.601697 | 0.675077 | No |

El modelo seleccionado utilizó `C=1.0`, `solver="lbfgs"`, `max_iter=2000`, `random_state=42` y ninguna ponderación de clases. Frente al dummy, mejoró average precision en 0.041420, equivalente a 37.49%.

Posteriormente se evaluaron Extra Trees y histogram gradient boosting mediante configuraciones bloqueadas y folds temporales secuenciales sobre las semanas 4-6.

![Comparación de average precision](report_model_average_precision.png)

| Modelo | AP semana 7 | ROC-AUC | Log loss | AP media en CV temporal |
|---|---:|---:|---:|---:|
| Extra Trees | 0.153689 | 0.602314 | 0.341647 | 0.155191 |
| Regresión logística | 0.151899 | 0.600006 | 0.342382 | 0.154996 |
| Histogram gradient boosting | 0.145960 | 0.595577 | 0.343728 | 0.150105 |

Extra Trees obtuvo la mejor estimación puntual, pero su ganancia de AP fue solamente 0.001790. El intervalo bootstrap de 95% fue de -0.011327 a 0.014929 e incluyó el cero. Los intervalos de la ganancia en ROC-AUC y de la reducción de log loss también incluyeron cero.

Se conservó la regresión logística sin ponderación por ofrecer desempeño prácticamente equivalente, mayor interpretabilidad, implementación más sencilla y ninguna desventaja respaldada estadísticamente.

---

## 8. Selección del umbral

Las probabilidades de validación se encontraban aproximadamente entre 0.0283 y 0.4499, por lo que un umbral de 0.5 no producía ninguna predicción positiva. Se evaluaron todos los umbrales que modificaban una decisión, generando 3,865 candidatos.

La política maximizó F1 sobre la semana 7 y, en caso de empate, seleccionó el umbral más alto. Antes de abrir la semana 8 se bloqueó permanentemente:

```text
0.12485514806532311
```

| Multiplicador | Umbral | Precision | Recall | F1 | Specificity | Tasa de predicciones positivas |
|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.099884 | 13.264% | 74.707% | 0.225282 | 39.325% | 62.225% |
| 0.90 | 0.112370 | 13.908% | 61.593% | 0.226920 | 52.647% | 48.926% |
| 0.95 | 0.118612 | 14.686% | 56.440% | 0.233075 | 59.279% | 42.458% |
| **1.00, seleccionado** | **0.124855** | **15.484%** | **50.585%** | **0.237102** | **65.707%** | **36.093%** |
| 1.05 | 0.131098 | 15.509% | 42.857% | 0.227754 | 71.001% | 30.530% |
| 1.10 | 0.137341 | 15.314% | 34.895% | 0.212857 | 76.033% | 25.175% |
| 1.20 | 0.149826 | 14.556% | 23.419% | 0.179533 | 82.926% | 17.775% |

Los umbrales más bajos detectan más presiones, pero generan más alertas falsas. Los más altos reducen las alertas, pero omiten más presiones reales. Con el umbral seleccionado, la semana 7 produjo 2,259 verdaderos negativos, 1,179 falsos positivos, 211 falsos negativos y 216 verdaderos positivos.

---

## 9. Evaluación final fuera de tiempo

Antes de acceder a la semana 8 se verificaron las huellas del modelo y la política, el orden de las variables, los hiperparámetros, el umbral y la receta de entrenamiento. Se exportó un manifiesto pre-test y se prohibió cualquier ajuste posterior de modelo, variables, calibración o umbral.

El pipeline final se creó clonando el modelo seleccionado en la Etapa 7 y se ajustó exactamente una vez con las semanas 1-7: 31,815 filas, 3,710 positivos y 51 predictores. Ninguna fila de la semana 8 participó en el entrenamiento.

| Métrica | Validación semana 7 | Test semana 8 | Test menos validación |
|---|---:|---:|---:|
| Average precision | 0.151899 | 0.157750 | +0.005851 |
| ROC-AUC | 0.600006 | 0.596890 | -0.003117 |
| Log loss | 0.342382 | 0.347812 | +0.005430 |
| Precision | 0.154839 | 0.158940 | +0.004102 |
| Recall | 0.505855 | 0.476190 | -0.029664 |
| F1 | 0.237102 | 0.238332 | +0.001230 |
| Specificity | 0.657068 | 0.677665 | +0.020597 |
| Tasa de predicciones positivas | 0.360931 | 0.339784 | -0.021147 |
| Tasa real de positivos | 0.110479 | 0.113411 | +0.002933 |

Las métricas probabilísticas cambiaron de manera modesta. Esto respalda estabilidad temporal, aunque confirma que la discriminación absoluta sigue siendo limitada.

### 9.1 Matriz de confusión final

![Matriz de confusión del test final](report_test_confusion_matrix.png)

| Resultado | Conteo |
|---|---:|
| Verdaderos negativos | 2,670 |
| Falsos positivos | 1,270 |
| Falsos negativos | 264 |
| Verdaderos positivos | 240 |

El modelo detectó 240 de 504 presiones reales, equivalente a un recall de 47.619%. Solamente 15.894% de las predicciones positivas fueron correctas. Se predijo presión para 33.978% de los pass rushers, aproximadamente tres veces la tasa real de 11.341%.

### 9.2 Ranking y calibración

![Estabilidad de deciles entre validación y test](stage9_decile_stability.png)

| Dataset | Tasa real | Probabilidad media | Brier score | Tasa del decil superior | Lift del decil superior |
|---|---:|---:|---:|---:|---:|
| Validación semana 7 | 0.110479 | 0.117648 | 0.097418 | 0.170543 | 1.543670 |
| Test semana 8 | 0.113411 | 0.115319 | 0.099395 | 0.191011 | 1.684234 |

En la semana 8, average precision fue 1.391 veces la prevalencia y el decil con mayor puntuación alcanzó una tasa de presión de 19.101%. La probabilidad media, 11.532%, fue cercana a la tasa real de 11.341%. El ranking probabilístico es, por lo tanto, más útil que la clasificación binaria por sí sola.

---

## 10. Hallazgos principales

1. **Existe señal, pero es modesta.** Las variables del snap mejoraron el ranking frente al baseline, aunque el ROC-AUC final permaneció por debajo de 0.60 y precision por debajo de 16%.
2. **El modelo sencillo estaba justificado.** Extra Trees no demostró una mejora estable sobre la regresión logística.
3. **La generalización temporal fue estable.** Las métricas de validación y test permanecieron cercanas y no ocurrió un colapso en la semana 8.
4. **El umbral favorece sensibilidad.** Recupera casi la mitad de las presiones, pero genera numerosas alertas falsas.
5. **El ranking es el uso más sólido.** Los grupos de mayor puntuación concentran más presiones que la población completa.

---

## 11. Limitaciones y riesgos

### 11.1 Alcance temporal limitado

Los datos cubren únicamente las semanas 1-8. El desempeño podría cambiar más adelante en la temporada, en playoffs o en otra temporada.

### 11.2 Dependencia de las etiquetas

El target hereda cualquier subjetividad, regla de cobertura o error presente en las etiquetas de scouting PFF.

### 11.3 Información exclusiva del snap

La presión depende de eventos posteriores, como la ejecución del bloqueo, el movimiento del quarterback, el desarrollo de las rutas y las interacciones entre jugadores. Estos elementos se excluyeron deliberadamente para conservar el momento de predicción.

### 11.4 Observaciones correlacionadas

Varios pass rushers pueden pertenecer a una misma jugada. La separación temporal impide que una jugada aparezca en distintas particiones, pero las filas de una misma jugada no son estadísticamente independientes. Una extensión podría utilizar remuestreo agrupado por jugada.

### 11.5 Objetivo del umbral

F1 no representa un costo específico de scouting o negocio. Un umbral de producción debería responder a costos operativos explícitos y seleccionarse con nuevos datos de desarrollo, nunca modificarse después de revisar la semana 8.

### 11.6 Precision y validez externa limitadas

La mayoría de las clasificaciones positivas fueron falsos positivos. Además, las variables, el tracking y los códigos PFF pertenecen al dataset suministrado. Utilizar el modelo en otra fuente exigiría nuevas validaciones de esquema, etiquetas, unidades, calibración y tiempo.

### 11.7 Ausencia de interpretación causal

Los coeficientes, diferencias de variables y tasas por alineación representan asociaciones. No demuestran que cambiar la posición o alineación de un jugador causaría una presión.

---

## 12. Reproducibilidad y controles de calidad

El entorno final utilizó Python 3.14.0, NumPy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0 y joblib 1.5.3. Se utilizó `random_state=42` donde correspondía.

| Notebook | Propósito |
|---|---|
| [`01_data_validation.ipynb`](../notebooks/01_data_validation.ipynb) | Inventario, validación de esquemas, target y exportaciones intermedias |
| [`02_analytical_table.ipynb`](../notebooks/02_analytical_table.ipynb) | Ingeniería de variables al snap y validación de la tabla analítica |
| [`03_exploratory_analysis.ipynb`](../notebooks/03_exploratory_analysis.ipynb) | Análisis exploratorio y separación temporal |
| [`04_modeling_baselines.ipynb`](../notebooks/04_modeling_baselines.ipynb) | Modelos dummy y regresión logística |
| [`05_advanced_modeling.ipynb`](../notebooks/05_advanced_modeling.ipynb) | Modelos avanzados, CV temporal y bootstrap |
| [`06_threshold_selection.ipynb`](../notebooks/06_threshold_selection.ipynb) | Selección del umbral con validación y bloqueo de la política |
| [`07_final_evaluation.ipynb`](../notebooks/07_final_evaluation.ipynb) | Entrenamiento final controlado y evaluación única de la semana 8 |

La ejecución final limpia confirmó 14 de 14 marcadores PASS, 93 de 93 controles de calidad, 12 de 12 huellas de artefactos, conteos de ejecución secuenciales, cero errores y ningún ajuste posterior al test. El estado final del manifiesto fue `final_test_evaluation_completed_without_post_test_adjustment`.

| Artefacto | Propósito |
|---|---|
| [`stage7_selected_model.joblib`](../models/stage7_selected_model.joblib) | Modelo seleccionado antes de ajustar el umbral |
| [`stage8_selected_policy.json`](stage8_selected_policy.json) | Modelo, variables, umbral y protocolo de entrenamiento bloqueados |
| [`stage9_final_model.joblib`](../models/stage9_final_model.joblib) | Modelo ajustado una vez con las semanas 1-7 |
| [`stage9_pretest_manifest.json`](stage9_pretest_manifest.json) | Evidencia registrada antes del acceso al test |
| [`stage9_test_predictions.parquet`](stage9_test_predictions.parquet) | Probabilidades y decisiones finales por fila |
| [`stage9_test_metrics.csv`](stage9_test_metrics.csv) | Métricas finales |
| [`stage9_final_quality_report.csv`](stage9_final_quality_report.csv) | Los 93 controles finales |
| [`stage9_final_manifest.json`](stage9_final_manifest.json) | Inventario final y huellas de artefactos |
| [`report_visualizations.py`](../src/nfl_pressure/report_visualizations.py) | Generación reproducible de figuras |

---

## 13. Conclusión

El proyecto produjo un pipeline validado y evaluado temporalmente para estimar la probabilidad de presión del pass rush al momento del snap.

La evidencia muestra que las variables de tracking contienen una señal real pero limitada; la regresión logística transparente funciona prácticamente igual que las alternativas no lineales; el desempeño permaneció estable de la semana 7 a la semana 8; y el umbral bloqueado recupera casi la mitad de las presiones, aunque genera numerosos falsos positivos.

El modelo debe posicionarse como una herramienta de priorización analítica. Puede ordenar observaciones para revisión, enriquecer el análisis de futbol americano o aportar una entrada a un sistema más amplio con contexto post-snap y experiencia de dominio.

Lo más importante es que el resultado final se obtuvo mediante un protocolo predefinido y auditable. No se modificó el modelo, las variables, los hiperparámetros, el umbral ni la calibración después de acceder a los datos de test.

---

## Apéndice A. Definición de métricas

| Métrica | Interpretación práctica |
|---|---|
| Average precision | Qué tan bien las probabilidades altas concentran las presiones poco frecuentes |
| ROC-AUC | Probabilidad de que una presión reciba mayor puntuación que un caso sin presión |
| Log loss | Penalización de probabilidades incorrectas y demasiado confiadas; menor es mejor |
| Precision | Proporción de presiones predichas que fueron presiones reales |
| Recall | Proporción de presiones reales detectadas por el umbral |
| F1 | Media armónica entre precision y recall |
| Specificity | Proporción de casos sin presión clasificados correctamente |
| Brier score | Diferencia cuadrática media entre probabilidad y resultado; menor es mejor |
| Lift | Tasa de presión de un grupo dividida entre la prevalencia general |

## Apéndice B. Resumen del test final

| Elemento | Valor final bloqueado |
|---|---|
| Modelo seleccionado | `LogisticRegression_unweighted` |
| Periodo final de entrenamiento | Semanas 1-7 |
| Periodo de test | Semana 8 |
| Predictores | 51 |
| Umbral de clasificación | `0.12485514806532311` |
| Filas de semana 8 | 4,444 |
| Jugadas de semana 8 | 1,031 |
| Positivos de semana 8 | 504 |
| Average precision | 0.157750 |
| ROC-AUC | 0.596890 |
| Log loss | 0.347812 |
| F1 | 0.238332 |
| Lift del decil superior | 1.684234 |
| Ajustes posteriores al test | Ninguno |
