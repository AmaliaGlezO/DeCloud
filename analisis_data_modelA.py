import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pandas as pd
#import cv2

# Configuración
BASE_PATH = "data/cloud_detection/Train/"
IMAGES_DIR = os.path.join(BASE_PATH, "masked")  # Imágenes originales
MASKS_DIR = os.path.join(BASE_PATH, "overall-mask")  # Máscaras de segmentación

def check_directory_structure():
    """Verifica la estructura de directorios"""
    print("=== Verificando estructura de directorios ===")
    print(f"Ruta base: {BASE_PATH}")
    print(f"Directorio de imágenes: {IMAGES_DIR}")
    print(f"Directorio de máscaras: {MASKS_DIR}")
    print(f"¿Existe imágenes?: {os.path.exists(IMAGES_DIR)}")
    print(f"¿Existe máscaras?: {os.path.exists(MASKS_DIR)}")
    
    if os.path.exists(IMAGES_DIR):
        images = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.tif')]
        print(f"\nNúmero de imágenes: {len(images)}")
        
    if os.path.exists(MASKS_DIR):
        masks = [f for f in os.listdir(MASKS_DIR) if f.endswith('.tif')]
        print(f"Número de máscaras: {len(masks)}")

def load_tif_image(filepath):
    """Carga una imagen TIFF usando rasterio"""
    with rasterio.open(filepath) as src:
        image = src.read()  # Lee todas las bandas
        metadata = {
            'shape': image.shape,
            'dtype': image.dtype,
            'count': src.count,
            'crs': src.crs,
            'transform': src.transform
        }
    return image, metadata

def analyze_images():
    """Realiza análisis detallado de las imágenes y máscaras"""
    
    # Obtener lista de archivos
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith('.tif')])
    mask_files = sorted([f for f in os.listdir(MASKS_DIR) if f.endswith('.tif')])
    
    print(f"\n=== Análisis de Datos ===")
    print(f"Total de imágenes: {len(image_files)}")
    print(f"Total de máscaras: {len(mask_files)}")
    
    # Verificar correspondencia
    common_files = set(image_files) & set(mask_files)
    print(f"Archivos coincidentes: {len(common_files)}")
    
    if len(common_files) == 0:
        print("¡Advertencia: No hay archivos coincidentes entre imágenes y máscaras!")
        print("Ejemplos de imágenes:", image_files[:5])
        print("Ejemplos de máscaras:", mask_files[:5])
    
    # Análisis de las primeras 5 imágenes
    print("\n=== Análisis de muestras individuales ===")
    
    stats_data = []
    
    for i in range(min(5, len(image_files))):
        img_path = os.path.join(IMAGES_DIR, image_files[i])
        mask_path = os.path.join(MASKS_DIR, mask_files[i])
        
        # Cargar imagen
        img, img_meta = load_tif_image(img_path)
        mask, mask_meta = load_tif_image(mask_path)
        
        print(f"\n--- Imagen {i+1}: {image_files[i]} ---")
        print(f"  Imagen - Shape: {img.shape}, Dtype: {img.dtype}, Bandas: {img_meta['count']}")
        print(f"  Máscara - Shape: {mask.shape}, Dtype: {mask.dtype}, Bandas: {mask_meta['count']}")
        
        # Estadísticas
        img_stats = {
            'filename': image_files[i],
            'image_shape': img.shape,
            'image_dtype': str(img.dtype),
            'image_min': float(img.min()),
            'image_max': float(img.max()),
            'image_mean': float(img.mean()),
            'image_std': float(img.std()),
            'mask_shape': mask.shape,
            'mask_dtype': str(mask.dtype),
            'mask_min': float(mask.min()),
            'mask_max': float(mask.max()),
            'mask_mean': float(mask.mean()),
            'unique_mask_values': np.unique(mask).tolist()
        }
        stats_data.append(img_stats)
    
    # Análisis estadístico completo
    print("\n=== Análisis Estadístico Completo ===")
    
    all_images = []
    all_masks = []
    
    for img_file, mask_file in tqdm(zip(image_files[:50], mask_files[:50]), 
                                   total=min(50, len(image_files)), 
                                   desc="Procesando muestra"):
        img_path = os.path.join(IMAGES_DIR, img_file)
        mask_path = os.path.join(MASKS_DIR, mask_file)
        
        img, _ = load_tif_image(img_path)
        mask, _ = load_tif_image(mask_path)
        
        all_images.append(img)
        all_masks.append(mask)
    
    # Convertir a arrays
    all_images = np.array(all_images)
    all_masks = np.array(all_masks)
    
    print(f"\nDimensiones del dataset de muestra:")
    print(f"Imágenes: {all_images.shape}")
    print(f"Máscaras: {all_masks.shape}")
    
    print(f"\nEstadísticas de imágenes:")
    print(f"  Rango: [{all_images.min():.2f}, {all_images.max():.2f}]")
    print(f"  Media: {all_images.mean():.2f} ± {all_images.std():.2f}")
    
    print(f"\nEstadísticas de máscaras:")
    print(f"  Rango: [{all_masks.min():.2f}, {all_masks.max():.2f}]")
    print(f"  Media: {all_masks.mean():.2f} ± {all_masks.std():.2f}")
    
    # Distribución de clases en máscaras
    unique_values, counts = np.unique(all_masks, return_counts=True)
    print(f"\nDistribución de clases en máscaras:")
    for val, count in zip(unique_values, counts):
        percentage = (count / all_masks.size) * 100
        print(f"  Clase {int(val)}: {count:,} píxeles ({percentage:.2f}%)")
    
    # Análisis por bandas (si es imagen multiespectral)
    if all_images.shape[1] > 1:  # Si tiene múltiples bandas
        print(f"\n=== Análisis por Bandas ===")
        for b in range(min(5, all_images.shape[1])):  # Primeras 5 bandas
            band_data = all_images[:, b, :, :]
            print(f"Banda {b+1}: Media={band_data.mean():.2f}, "
                  f"Std={band_data.std():.2f}, "
                  f"Range=[{band_data.min():.2f}, {band_data.max():.2f}]")
    
    return stats_data, all_images, all_masks

def visualize_samples(images, masks, n_samples=3):
    """Visualiza muestras del dataset"""
    
    print(f"\n=== Visualización de {n_samples} muestras ===")
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        img = images[i]
        mask = masks[i]
        
        # Para imágenes multibanda, mostrar composición RGB si es posible
        if img.shape[0] >= 3:  # Al menos 3 bandas
            # Normalizar para visualización
            img_rgb = img[:3]  # Tomar primeras 3 bandas
            img_rgb = np.transpose(img_rgb, (1, 2, 0))
            
            # Normalización por banda
            for b in range(3):
                min_val = img_rgb[:,:,b].min()
                max_val = img_rgb[:,:,b].max()
                if max_val > min_val:
                    img_rgb[:,:,b] = (img_rgb[:,:,b] - min_val) / (max_val - min_val)
        else:
            img_rgb = img[0] if len(img.shape) == 3 else img
            img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min())
        
        # Mostrar imagen
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f'Imagen {i+1}')
        axes[i, 0].axis('off')
        
        # Mostrar máscara
        mask_display = mask[0] if len(mask.shape) == 3 else mask
        axes[i, 1].imshow(mask_display, cmap='gray')
        axes[i, 1].set_title(f'Máscara {i+1}')
        axes[i, 1].axis('off')
        
        # Mostrar superposición
        axes[i, 2].imshow(img_rgb)
        axes[i, 2].imshow(mask_display, alpha=0.5, cmap='Reds')
        axes[i, 2].set_title(f'Superposición {i+1}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Histogramas
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histograma de valores de imagen
    axes[0].hist(images[:5].flatten(), bins=50, alpha=0.7, color='blue')
    axes[0].set_title('Distribución de valores de píxeles (imágenes)')
    axes[0].set_xlabel('Valor')
    axes[0].set_ylabel('Frecuencia')
    
    # Histograma de valores de máscara
    axes[1].hist(masks[:5].flatten(), bins=50, alpha=0.7, color='red')
    axes[1].set_title('Distribución de valores de píxeles (máscaras)')
    axes[1].set_xlabel('Valor')
    axes[1].set_ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.show()

def generate_summary_report(stats_data):
    """Genera un reporte resumen del análisis"""
    
    print("\n" + "="*60)
    print("REPORTE DE ANÁLISIS DE DATOS PARA SEGMENTACIÓN DE NUBES")
    print("="*60)
    
    df_stats = pd.DataFrame(stats_data)
    
    print("\n1. RESUMEN GENERAL:")
    print(f"   • Total de muestras analizadas: {len(df_stats)}")
    print(f"   • Formato de imágenes: TIFF")
    
    print("\n2. DIMENSIONES:")
    unique_shapes = df_stats['image_shape'].unique()
    print(f"   • Formas únicas de imágenes: {len(unique_shapes)}")
    for shape in unique_shapes:
        count = (df_stats['image_shape'] == shape).sum()
        print(f"     - {shape}: {count} imágenes")
    
    print("\n3. RANGOS DE VALORES:")
    print(f"   • Imágenes: Min={df_stats['image_min'].min():.2f}, "
          f"Max={df_stats['image_max'].max():.2f}")
    print(f"   • Máscaras: Min={df_stats['mask_min'].min():.2f}, "
          f"Max={df_stats['mask_max'].max():.2f}")
    
    print("\n4. RECOMENDACIONES PARA EL MODELO:")
    
    # Verificar balance de clases
    mask_values = set()
    for vals in df_stats['unique_mask_values']:
        mask_values.update(vals)
    
    print(f"   • Clases únicas en máscaras: {sorted(list(mask_values))}")
    
    if len(mask_values) == 2:
        print("   • Binario: Usar sigmoid en la capa final")
    else:
        print("   • Multiclase: Usar softmax en la capa final")
    
    # Recomendaciones de preprocesamiento
    print("\n5. PREPROCESAMIENTO SUGERIDO:")
    print("   • Normalizar imágenes según estadísticas calculadas")
    print("   • Verificar correspondencia imagen-máscara")
    print("   • Considerar aumento de datos si el dataset es pequeño")
    
    return df_stats

def main():
    """Función principal"""
    
    # 1. Verificar estructura
    check_directory_structure()
    
    # 2. Cargar y analizar datos
    stats_data, sample_images, sample_masks = analyze_images()
    
    # 3. Visualizar muestras
    if len(sample_images) > 0:
        visualize_samples(sample_images, sample_masks, n_samples=min(3, len(sample_images)))
    
    # 4. Generar reporte
    df_stats = generate_summary_report(stats_data)
    
    # 5. Guardar resultados
    output_dir = "data_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    df_stats.to_csv(os.path.join(output_dir, "image_statistics.csv"), index=False)
    
    print(f"\nAnálisis completado. Resultados guardados en: {output_dir}/")
    
    return sample_images, sample_masks, df_stats

if __name__ == "__main__":
    # Instalar dependencias si es necesario
    try:
        import rasterio
    except ImportError:
        print("Instalando dependencias necesarias...")
        import subprocess
        subprocess.check_call(["pip", "install", "rasterio", "matplotlib", 
                               "seaborn", "pandas", "tqdm", "opencv-python"])
    
    # Ejecutar análisis
    images, masks, stats = main()