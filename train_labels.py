import os
import csv
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DATA_URL = (
    "https://data.source.coop/radiantearth/"
    "cloud-cover-detection-challenge/final/public"
)
OUT_DIR = "train_labels"
CSV_FILE = "train_metadata.csv"

BANDS = ["B02.tif", "B03.tif", "B04.tif", "B08.tif", "B11.tif", "B12.tif"]
MAX_WORKERS = 16  # ajusta según tu ancho de banda / CPU

os.makedirs(OUT_DIR, exist_ok=True)

with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Preconstruimos la lista de trabajos (url, destino)
tasks = []
for row in rows:
    chip_id = row["chip_id"]
    cloudpath = row["cloudpath"].replace("az://./", "").lstrip("/")  
    chip_out = os.path.join(OUT_DIR, chip_id)
    os.makedirs(chip_out, exist_ok=True)

    label_url = f"{BASE_DATA_URL}/train_labels/{chip_id}.tif"
    label_out = os.path.join(chip_out, "label.tif")
    if not os.path.exists(label_out):
        tasks.append((label_url, label_out))


def download_one(args):
    url, out_path = args
    try:
        r = requests.get(url, stream=True, timeout=60)
    except requests.RequestException as e:
        return f"ERROR_CONN {url} {e}"

    if r.status_code == 200:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as fp:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    fp.write(chunk)
        return f"OK {url}"
    else:
        return f"HTTP_{r.status_code} {url}"


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = [ex.submit(download_one, t) for t in tasks]
    for f in tqdm(as_completed(futures), total=len(futures), desc="Descargando chips"):
        msg = f.result()
        if not msg.startswith("OK"):
            print(msg)








