# Generación de Música Mediante Modelado de Caracteres con Redes Neuronales Recurrentes (LSTM)

---

## Introducción Técnica
Este proyecto desarrolla un sistema de aprendizaje profundo enfocado en el modelado de secuencias musicales utilizando la arquitectura Long Short-Term Memory (LSTM). El objetivo es tratar la notación musical en formato ABC como un lenguaje estructurado, permitiendo que la red neuronal aprenda dependencias temporales, ritmos y estructuras melódicas a través de la predicción probabilística del siguiente carácter en una secuencia dada.

---

## Arquitectura del Sistema

El modelo está diseñado sobre una red neuronal recurrente multicapa con las siguientes especificaciones técnicas:

1. Capas Recurrentes: Tres capas de celdas LSTM apiladas para permitir la captura de patrones complejos en diferentes niveles de abstracción.
2. Dimensión Oculta: 512 unidades ocultas por cada capa, proporcionando la capacidad necesaria para memorizar secuencias de largo alcance sin incurrir en una degradación excesiva del gradiente.
3. Regularización: Implementación de una capa Dropout con una probabilidad de 0.5 entre las capas recurrentes para prevenir el sobreajuste y mejorar la capacidad de generalización del modelo.
4. Capa de Salida: Una capa lineal totalmente conectada que proyecta la salida de las LSTM hacia la dimensión total del vocabulario (caracteres únicos).



---

## Pipeline de Datos y Procesamiento

El flujo de datos se divide en etapas críticas para asegurar la compatibilidad con el entrenamiento en GPU:

1. Ingesta de Datos: Lectura del dataset songs (1).txt en formato de texto plano.
2. Definición del Vocabulario: Identificación de todos los caracteres únicos presentes en el corpus para construir los diccionarios de codificación (encoder) y decodificación (decoder).
3. Transformación Vectorial: Implementación de One-Hot Encoding para convertir los índices numéricos en vectores binarios procesables por las funciones de activación de la red.
4. Organización de Lotes (Batching): División del dataset en fragmentos de 100 caracteres de longitud, organizados en lotes de 128 muestras para maximizar el paralelismo durante el entrenamiento.

---

## Configuración del Entrenamiento

Para el proceso de optimización de los pesos de la red, se definieron los siguientes hiperparámetros y estrategias:

1. Función de Pérdida: CrossEntropyLoss (Entropía Cruzada), adecuada para problemas de clasificación multiclase donde cada carácter representa una categoría.
2. Algoritmo de Optimización: Adam (Adaptive Moment Estimation) con una tasa de aprendizaje constante de 0.001.
3. Estabilización de Gradientes: Aplicación de Gradient Clipping con un umbral de 5.0 para mitigar el problema de los gradientes explosivos, garantizando la estabilidad numérica durante las 100 épocas de entrenamiento.
4. Hardware: Ejecución optimizada mediante el uso de aceleración por hardware CUDA en entornos con GPU NVIDIA T4.



---

## Motor de Inferencia y Generación

La generación de nuevas piezas musicales no es determinista, sino que utiliza una lógica de muestreo probabilístico:

1. Semilla Inicial: El modelo recibe una cadena de texto (Seed) como punto de partida para establecer el contexto inicial de la memoria.
2. Muestreo Top-K: Se implementó un algoritmo de selección que filtra las K predicciones más probables del modelo. Esto permite un equilibrio entre la coherencia estructural (alta probabilidad) y la creatividad melódica (variabilidad).
3. Normalización: Las salidas de la capa Softmax se normalizan nuevamente después de la selección Top-K para asegurar que la distribución de probabilidad sea válida para la función de elección aleatoria de NumPy.
4. Robustez de Dimensiones: El sistema incluye un tratamiento de datos mediante el método flatten() para asegurar la compatibilidad de arreglos entre PyTorch y NumPy durante la predicción iterativa.



---

## Requerimientos de Instalación

Para replicar el entorno de ejecución, se deben configurar los siguientes componentes:

Dependencias de sistema operativo:
- abcmidi: Herramientas para el procesamiento de archivos en formato ABC.
- timidity: Sintetizador de audio para la conversión de MIDI a WAV.

Bibliotecas de Python necesarias:
- torch (v1.x o superior)
- numpy
- matplotlib
- mitdeeplearning
- IPython

---

## Flujo de Conversión a Audio

Una vez que la red genera una secuencia de caracteres coherente, el proceso de salida sigue estos pasos:
1. Identificación de patrones mediante expresiones regulares para extraer fragmentos de canciones válidos.
2. Generación de un archivo MIDI temporal a partir del texto generado.
3. Síntesis de la señal de audio mediante el procesamiento de la tabla de ondas.
4. Visualización y reproducción de la forma de onda directamente en la interfaz de usuario.



---
