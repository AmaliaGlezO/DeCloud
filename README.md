

# Cloud Detection & Cloud Removal 

**Dataset:** 38-Cloud (Landsat 8)
**Objetivo:** Detección precisa de nubes + reconstrucción visual de regiones cubiertas

---

## Diagnóstico de datos

### Limitación fundamental del dataset

El dataset **NO contiene imágenes limpias del mismo lugar** sin nubes.
Por tanto:

* No se puede entrenar *cloud removal supervisado clásico*
* No existe ground truth real “sin nubes”

### Posible Solución

Separar el problema en **dos tareas**:

1. **Detección de nubes (segmentación)**
2. **Reconstrucción (inpainting) condicionada por máscara**

Esto evita falsos supuestos y mantiene coherencia física.

---

## Visión general del pipeline

```
[R, G, B, NIR]  ──────►  Cloud Detector (Attention U-Net)
                               │
                               ▼
                        Cloud Mask (0/1)
                               │
                               ▼
[R, G, B, NIR, Mask] ─►  Cloud Inpainting U-Net
                               │
                               ▼
                 Imagen reconstruida (RGB / RGB+NIR)
```

**Dos modelos separados, entrenados con objetivos distintos**

---

## MODELO 1 — Cloud Detection

### Objetivo

Detectar **nubes a nivel de píxel** con alta precisión.

### Input

* 4 canales:

  * Red (B4)
  * Green (B3)
  * Blue (B2)
  * NIR (B5)

```
Input shape: (4, 384, 384)
```

### Output

* Máscara binaria:

```
Output shape: (1, 384, 384)
0 = no nube
1 = nube
```

---

### Arquitectura

**Attention U-Net**

* Encoder–decoder
* Skip connections con **attention gates**
* Mejor detección de:

  * nubes delgadas
  * bordes suaves
  * confusión con nieve / bruma

Arquitectura probada en medical imaging → perfecta para nubes.

---

### Loss function

```
Loss = BCE + Dice
```

* BCE: estabilidad
* Dice: penaliza falsos negativos (nubes finas)

---

### Métricas

* IoU
* Dice coefficient
* Precision / Recall
* Visualización de máscaras superpuestas

---

### Output del modelo

* Modelo guardado por epoch
* Mejor modelo según IoU validación

---
### Citados 2 papers que hacen lo mismo
```  
@INPROCEEDINGS{38-cloud-1,
  author={S. {Mohajerani} and P. {Saeedi}},
  booktitle={IGARSS 2019 - 2019 IEEE International Geoscience and Remote Sensing Symposium},
  title={Cloud-Net: An End-To-End Cloud Detection Algorithm for Landsat 8 Imagery},
  year={2019},
  volume={},
  number={},
  pages={1029-1032},
  doi={10.1109/IGARSS.2019.8898776},
  ISSN={2153-6996},
  month={July},
}

@INPROCEEDINGS{38-cloud-2,   
  author={S. Mohajerani and T. A. Krammer and P. Saeedi},   
  booktitle={2018 IEEE 20th International Workshop on Multimedia Signal Processing (MMSP)},   
  title={{"A Cloud Detection Algorithm for Remote Sensing Images Using Fully Convolutional Neural Networks"}},   
  year={2018},    
  pages={1-5},   
  doi={10.1109/MMSP.2018.8547095},   
  ISSN={2473-3628},   
  month={Aug},  
}
```

## MODELO 2 — Cloud Removal (Inpainting condicional)

### Objetivo

Reconstruir regiones cubiertas por nubes **sin ground truth limpio**, usando aprendizaje auto-supervisado.

---

## Idea clave 

**No intentamos quitar nubes reales durante el entrenamiento**

En su lugar:

1. Usamos **zonas SIN nubes**
2. Generamos **máscaras artificiales**
3. Entrenamos al modelo a reconstruirlas
4. En inferencia, aplicamos el modelo a nubes reales

📌 Esto es *self-supervised inpainting*.

---

### Input

```
[R, G, B, NIR, Mask]
```

* Mask = 1 → zona a reconstruir
* Mask = 0 → zona válida

```
Input shape: (5, 384, 384)
```

---

### Output

* Imagen reconstruida:

```
Output shape: (4, 384, 384)
```

---

### Arquitectura

**U-Net para inpainting**, con:

* Partial Convolutions **o**
* Attention en skip connections
* Normalización por máscara

El modelo **NO ve píxeles ocultos**.

---

### Loss física (solo en zona oculta)

```
Loss = L1_masked + SSIM_masked
```

Donde:

* L1 → fidelidad espectral
* SSIM → coherencia estructural
* Calculada **solo donde mask == 1**

Esto evita:

* copiar píxeles visibles
* blur innecesario

---

### Métricas visuales reales

No hay métricas clásicas de test, así que:

* Comparación visual
* Error L1 en zonas simuladas
* Evolución temporal por epoch
* GIFs de reconstrucción

---
# Modelo 3 Cloud Removal 
La idea de proponer este tercer modelo fue hacer comparaciones con la propuesta que tenía y con otra manera de reconstruir el terreno borrando las nubes en imágenes satelitales.

Para este modelo se usó otro dataset con pares de imágenes limpias y con nubes de las mismas zonas, y se propone una arquitectura U-Net modificada que toma como entrada de 4 canales: los 3 canales RGB de la imagen con nubes más un canal adicional de máscara binaria que identifica las regiones nubosas. Esta máscara se genera automáticamente mediante un modelo de segmentación preentrenado. La pérdida combinada (L1 total ponderada) prioriza la reconstrucción en las áreas nubladas mientras preserva las regiones ya limpias, aprovechando así información guiada para mejorar la remoción de nubes.

---
### Dentro de la carpeta models abrir Model_1_2_3.ipynb