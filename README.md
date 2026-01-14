# DeCloud

Este proyecto presenta un sistema basado en redes neuronales cuyo objetivo es eliminar nubes de imágenes satelitales y reconstruir, de la forma más realista posible, la información que queda oculta bajo ellas. Las imágenes utilizadas provienen del satélite Sentinel-2, que ofrece información multiespectral muy valiosa, pero que con frecuencia se ve afectada por la presencia de nubes, lo que limita su uso en tareas de análisis territorial, ambiental o agrícola.

## ¿Qué hace?
La idea principal del trabajo es abordar el problema en dos etapas (Model A, Model B). En lugar de intentar reconstruir directamente una imagen sin nubes, el sistema primero aprende a identificar con precisión qué partes de la imagen están cubiertas por nubes y, solo después, utiliza esa información para reconstruir las zonas ocultas. Este enfoque permite simplificar el aprendizaje de cada modelo y obtener resultados más estables.

## Model A
En la primera etapa se utiliza una red neuronal de segmentación que recibe como entrada una imagen satelital multibanda y produce como salida una máscara que indica, píxel a píxel, la presencia o ausencia de nubes. Para esta tarea se emplea una arquitectura tipo U-Net, muy utilizada en segmentación de imágenes, ya que permite capturar tanto el contexto global de la escena como los detalles finos gracias a su estructura de encoder-decoder y a las conexiones entre niveles. Este modelo no elimina las nubes, sino que aprende únicamente a localizarlas de forma precisa.

## Model B
En la segunda etapa, el sistema utiliza la imagen original junto con la máscara de nubes generada en la etapa anterior para reconstruir las zonas cubiertas. Este modelo aprende a rellenar esas áreas utilizando la información del entorno y las distintas bandas espectrales, generando una imagen completa que intenta ser coherente. Para esta reconstrucción se pueden emplear modelos generativos, como redes adversarias (GANs), o arquitecturas más recientes basadas en mecanismos de atención, como los Transformers. En ambos casos, el objetivo es que la imagen final se aproxime lo máximo posible a una imagen real sin nubes.


## ¿Para qué sirve?

- Este proyecto puede ser útil para cualquiera que trabaje con imágenes satelitales y necesite ver el terreno completo, incluso cuando hay nubes. Por ejemplo:

- Análisis de cultivos y zonas agrícolas

- Monitoreo de bosques y áreas naturales

- Estudio del crecimiento de ciudades

- Observación de zonas afectadas por desastres naturales