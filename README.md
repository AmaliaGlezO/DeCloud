# DeCloud

DeCloud es un proyecto que utiliza redes neuronales para eliminar las nubes de imágenes satelitales y poder ver el terreno que queda oculto debajo.

La idea principal es simple: cuando una nube tapa parte de una imagen, el modelo intenta reconstruir esa zona de la forma más realista posible.

## ¿Qué hace?

Cuando tienes una imagen satelital con nubes que no te dejan ver bien el suelo, DeCloud se encarga de “rellenar” esas partes faltantes.
Para lograrlo, usa una red neuronal entrenada con distintos tipos de paisajes (ciudades, bosques, agua, campos, etc.) y aprende cómo suelen verse esos terrenos.

Con esa información, el modelo predice qué podría haber debajo de las nubes y genera una nueva imagen sin ellas.

## ¿Para qué sirve?

- Este proyecto puede ser útil para cualquiera que trabaje con imágenes satelitales y necesite ver el terreno completo, incluso cuando hay nubes. Por ejemplo:

- Análisis de cultivos y zonas agrícolas

- Monitoreo de bosques y áreas naturales

- Estudio del crecimiento de ciudades

- Observación de zonas afectadas por desastres naturales