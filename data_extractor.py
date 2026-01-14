import tarfile
import os

# Ruta del archivo .tar.gz
archivo_tar = "ROIs1158_spring_s1.tar.gz"
# Directorio de destino
directorio_destino = "./extraido/"

# Crear directorio si no existe
os.makedirs(directorio_destino, exist_ok=True)

# Extraer el archivo
try:
    with tarfile.open(archivo_tar, "r:gz") as tar:
        tar.extractall(path=directorio_destino)
    print(f"Archivo extraído en: {directorio_destino}")
except Exception as e:
    print(f"Error al extraer: {e}")