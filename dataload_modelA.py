from pystac_client import Client
import planetary_computer
import xarray as xr
import rioxarray
import numpy as np
import rasterio
from pathlib import Path

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# Buscar colecciones relevantes por texto
for col in catalog.get_all_collections():
    if "Sentinel-2" in col.title or "Landsat" in col.title:
        print(col.id, col.title)

min_lon, min_lat, max_lon, max_lat = -123.5, 37.0, -121.5, 38.5
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[min_lon, min_lat, max_lon, max_lat],
    datetime="2020-01-01/2020-12-31",
    query={"eo:cloud_cover": {"gt": 20}}  # asegurar nubes
)

items = list(search.get_items())


# Buscar colecciones relevantes por texto
for col in catalog.get_all_collections():
    if "Sentinel-2" in col.title or "Landsat" in col.title:
        print(col.id, col.title)

min_lon, min_lat, max_lon, max_lat = -123.5, 37.0, -121.5, 38.5
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[min_lon, min_lat, max_lon, max_lat],
    datetime="2020-01-01/2020-12-31",
    query={"eo:cloud_cover": {"gt": 20}}  # asegurar nubes
)

items = list(search.get_items())

bbox = [-83.5, 22.0, -81.0, 23.5]  

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2020-01-01/2020-12-31",
    query={"eo:cloud_cover": {"gt": 20, "lt": 80}},  # escenas con nubes pero no totalmente nubosas
)
print(len(items))

def open_s2_rgb_scl(item):
    assets = item.assets

    # URLs firmadas para Planetary Computer
    href_b2 = planetary_computer.sign(assets["B02"].href)
    href_b3 = planetary_computer.sign(assets["B03"].href)
    href_b4 = planetary_computer.sign(assets["B04"].href)
    href_scl = planetary_computer.sign(assets["SCL"].href)

    # Cargamos cada banda como xarray/rioxarray
    b2 = rioxarray.open_rasterio(href_b2)  # shape (1, y, x)
    b3 = rioxarray.open_rasterio(href_b3)
    b4 = rioxarray.open_rasterio(href_b4)
    scl = rioxarray.open_rasterio(href_scl)

    # Asegurar misma resolución/extent (normalmente ya coinciden)
    # Stack RGB en un solo array: (3, y, x)
    rgb = xr.concat([b4, b3, b2], dim="band")
    rgb = rgb.assign_coords(band=["R", "G", "B"])

    return rgb, scl
# Valores de SCL que consideraremos como "nube"
SCL_CLOUD_VALUES = [3, 8, 9, 10]  # sombra + nubes

def scl_to_cloud_mask(scl):
    # scl: DataArray (1, y, x)
    scl_data = scl.squeeze().values  # (y, x)

    mask_cloud = np.isin(scl_data, SCL_CLOUD_VALUES).astype(np.uint8)  # 1 = nube, 0 = no nube

    # Lo devolvemos como xarray con misma georeferencia
    cloud_da = xr.DataArray(
        mask_cloud,
        coords={"y": scl.y, "x": scl.x},
        dims=("y", "x"),
    )
    cloud_da.rio.write_crs(scl.rio.crs, inplace=True)
    cloud_da.rio.write_transform(scl.rio.transform(), inplace=True)

    return cloud_da
def extract_patches(rgb, mask, patch_size=256, stride=256, min_cloud_fraction=0.01):
    """
    rgb: DataArray (band=3, y, x)
    mask: DataArray (y, x) con 0/1
    """
    patches = []
    H, W = mask.shape
    for y0 in range(0, H - patch_size + 1, stride):
        for x0 in range(0, W - patch_size + 1, stride):
            y1 = y0 + patch_size
            x1 = x0 + patch_size

            patch_img = rgb.isel(y=slice(y0, y1), x=slice(x0, x1))
            patch_mask = mask.isel(y=slice(y0, y1), x=slice(x0, x1))

            # Opcional: filtrar patches sin nubes
            cloud_frac = float(patch_mask.values.mean())
            if cloud_frac < min_cloud_fraction:
                continue

            patches.append((patch_img, patch_mask))
    return patches


out_dir = Path("cloud_dataset_s2")
(out_dir / "images").mkdir(parents=True, exist_ok=True)
(out_dir / "masks").mkdir(parents=True, exist_ok=True)

patch_id = 0
for item in items[:10]:  # por ejemplo, primeras 10 escenas
    rgb, scl = open_s2_rgb_scl(item)
    cloud_mask = scl_to_cloud_mask(scl)

    patches = extract_patches(rgb, cloud_mask, patch_size=256, stride=256, min_cloud_fraction=0.01)

    for img_da, m_da in patches:
        # img_da: (3, y, x), m_da: (y, x)
        img_path = out_dir / "images" / f"{patch_id:06d}.tif"
        mask_path = out_dir / "masks" / f"{patch_id:06d}.tif"

        # Guardar imagen RGB
        with rasterio.open(
            img_path,
            "w",
            driver="GTiff",
            height=img_da.sizes["y"],
            width=img_da.sizes["x"],
            count=3,
            dtype=img_da.dtype,
        ) as dst:
            dst.write(img_da.values)

        # Guardar máscara (una banda, 0/1)
        with rasterio.open(
            mask_path,
            "w",
            driver="GTiff",
            height=m_da.sizes["y"],
            width=m_da.sizes["x"],
            count=1,
            dtype=m_da.dtype,
        ) as dst:
            dst.write(m_da.values, 1)

        patch_id += 1
    print(f"Procesado item {item.id}, extraídos {len(patches)} patches.")