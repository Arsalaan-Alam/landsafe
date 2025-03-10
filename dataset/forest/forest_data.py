import requests
import rasterio
import os
import csv
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
TIF_DIR = os.path.join(BASE_DIR, "data", "forest_loss")  
os.makedirs(TIF_DIR, exist_ok=True)  

BASE_URL = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/"

def get_forest_loss(lat, lon, year):
    tile_lat = int(lat // 10) * 10
    tile_lon = int(lon // 10) * 10

    lat_suffix = "N" if tile_lat >= 0 else "S"
    lon_suffix = "E" if tile_lon >= 0 else "W"

    tile_lat = abs(tile_lat)
    tile_lon = abs(tile_lon)

    treecover_filename = f"Hansen_GFC-2023-v1.11_treecover2000_{tile_lat:02d}{lat_suffix}_{tile_lon:03d}{lon_suffix}.tif"
    treecover_path = os.path.join(TIF_DIR, treecover_filename)
    treecover_url = BASE_URL + treecover_filename

    if not os.path.exists(treecover_path):
        response = requests.get(treecover_url, stream=True)
        if response.status_code == 200:
            with open(treecover_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        else:
            return None

    lossyear_filename = f"Hansen_GFC-2023-v1.11_lossyear_{tile_lat:02d}{lat_suffix}_{tile_lon:03d}{lon_suffix}.tif"
    lossyear_path = os.path.join(TIF_DIR, lossyear_filename)
    lossyear_url = BASE_URL + lossyear_filename

    if not os.path.exists(lossyear_path):
        response = requests.get(lossyear_url, stream=True)
        if response.status_code == 200:
            with open(lossyear_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        else:
            return None

    try:
        with rasterio.open(treecover_path) as src:
            row, col = src.index(lon, lat)
            pixels_per_degree = 4000 / 10
            pixels_per_mile = pixels_per_degree / 69
            radius = int(5 * pixels_per_mile)
            window = ((max(0, row - radius), min(src.height, row + radius)), 
                      (max(0, col - radius), min(src.width, col + radius)))
            treecover_data = src.read(1, window=window)
            forest_threshold = 25
            has_forest = np.any(treecover_data >= forest_threshold)
            if not has_forest:
                return 0
        
        with rasterio.open(lossyear_path) as src:
            row, col = src.index(lon, lat)
            pixels_per_degree = 4000 / 10
            pixels_per_mile = pixels_per_degree / 69
            radius = int(5 * pixels_per_mile)
            window = ((max(0, row - radius), min(src.height, row + radius)), 
                      (max(0, col - radius), min(src.width, col + radius)))
            lossyear_data = src.read(1, window=window)
            year_code = year - 2000
            forest_loss_years = lossyear_data[(lossyear_data > 0) & (lossyear_data < year_code)]
            if len(forest_loss_years) > 0:
                return 1
            else:
                return 0
                
    except Exception as e:
        return None
