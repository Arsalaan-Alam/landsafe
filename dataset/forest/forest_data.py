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
    # Calculate tile coordinates
    tile_lat = int(lat // 10) * 10
    tile_lon = int(lon // 10) * 10

    lat_suffix = "N" if tile_lat >= 0 else "S"
    lon_suffix = "E" if tile_lon >= 0 else "W"

    tile_lat = abs(tile_lat)
    tile_lon = abs(tile_lon)

    # Use treecover2000 TIF file first to check if there was forest
    treecover_filename = f"Hansen_GFC-2023-v1.11_treecover2000_{tile_lat:02d}{lat_suffix}_{tile_lon:03d}{lon_suffix}.tif"
    treecover_path = os.path.join(TIF_DIR, treecover_filename)
    treecover_url = BASE_URL + treecover_filename

    # Download the treecover file if it doesn't exist
    if not os.path.exists(treecover_path):
        print(f"\n📥 Downloading {treecover_filename} from {treecover_url} ...")
        response = requests.get(treecover_url, stream=True)

        if response.status_code == 200:
            with open(treecover_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)

            file_size = os.path.getsize(treecover_path) / (1024 * 1024)
            print(f"✅ Downloaded: {treecover_filename} ({file_size:.2f} MB)")
        else:
            print(f"❌ Error: Failed to download {treecover_filename}. URL may not exist.")
            return None
    else:
        file_size = os.path.getsize(treecover_path) / (1024 * 1024)
        print(f"📂 File exists: {treecover_filename} ({file_size:.2f} MB)")

    # Now also download the lossyear file to check forest loss
    lossyear_filename = f"Hansen_GFC-2023-v1.11_lossyear_{tile_lat:02d}{lat_suffix}_{tile_lon:03d}{lon_suffix}.tif"
    lossyear_path = os.path.join(TIF_DIR, lossyear_filename)
    lossyear_url = BASE_URL + lossyear_filename

    if not os.path.exists(lossyear_path):
        print(f"\n📥 Downloading {lossyear_filename} from {lossyear_url} ...")
        response = requests.get(lossyear_url, stream=True)

        if response.status_code == 200:
            with open(lossyear_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)

            file_size = os.path.getsize(lossyear_path) / (1024 * 1024)
            print(f"✅ Downloaded: {lossyear_filename} ({file_size:.2f} MB)")
        else:
            print(f"❌ Error: Failed to download {lossyear_filename}. URL may not exist.")
            return None
    else:
        file_size = os.path.getsize(lossyear_path) / (1024 * 1024)
        print(f"📂 File exists: {lossyear_filename} ({file_size:.2f} MB)")

    try:
        # First check if there was significant tree cover in 2000
        with rasterio.open(treecover_path) as src:
            # Calculate the pixel coordinates for the center point
            row, col = src.index(lon, lat)
            
            # Define a radius for searching (approximately 5 miles)
            pixels_per_degree = 4000 / 10
            pixels_per_mile = pixels_per_degree / 69
            radius = int(5 * pixels_per_mile)
            
            # Create a window that covers the radius
            window_row_start = max(0, row - radius)
            window_row_stop = min(src.height, row + radius)
            window_col_start = max(0, col - radius)
            window_col_stop = min(src.width, col + radius)
            
            # Read the data within the window
            window = ((window_row_start, window_row_stop), (window_col_start, window_col_stop))
            treecover_data = src.read(1, window=window)
            
            # Tree cover percentage threshold (e.g., 25% or higher is considered forest)
            forest_threshold = 25
            
            # Check if the area had forest cover in 2000
            has_forest = np.any(treecover_data >= forest_threshold)
            
            print(f"🌳 Forest cover check: {'Forest detected' if has_forest else 'No significant forest'}")
            
            if not has_forest:
                print(f"🌳 No significant forest cover detected in 2000 at {lat}, {lon} ❌")
                return 0
        
        # Now check if there was forest loss before the landslide year
        with rasterio.open(lossyear_path) as src:
            # Calculate the pixel coordinates for the center point
            row, col = src.index(lon, lat)
            
            # Define a radius for searching (approximately 5 miles)
            pixels_per_degree = 4000 / 10
            pixels_per_mile = pixels_per_degree / 69
            radius = int(5 * pixels_per_mile)
            
            # Create a window that covers the radius
            window_row_start = max(0, row - radius)
            window_row_stop = min(src.height, row + radius)
            window_col_start = max(0, col - radius)
            window_col_stop = min(src.width, col + radius)
            
            # Read the data within the window
            window = ((window_row_start, window_row_stop), (window_col_start, window_col_stop))
            lossyear_data = src.read(1, window=window)
            
            print(f"🔍 Searching for forest loss within ~5 miles around ({lat}, {lon})")
            
            # Convert year to the format stored in the GFC dataset (year - 2000)
            year_code = year - 2000
            
            # Find forest loss years before the landslide
            forest_loss_years = lossyear_data[(lossyear_data > 0) & (lossyear_data < year_code)]
            
            if len(forest_loss_years) > 0:
                # There was forest loss before the landslide
                greatest_loss_year = np.max(forest_loss_years)
                actual_year = 2000 + int(greatest_loss_year)
                print(f"🌲 Forest Loss Detected in {actual_year} for {lat}, {lon} (Before landslide in {year}) ✅")
                return 1
            else:
                print(f"🌳 No Forest Loss Detected before {year} for {lat}, {lon} ❌")
                return 0
                
    except Exception as e:
        print(f"❌ Error processing coordinates {lat}, {lon}: {e}")
        return None

# test_locations = [
# (-8.2500, 123.1500, "04/04/21", 1),
#     (30.2937, 79.5603, "02/08/21", 1),
#     (21.654994, -158.061102, "03/09/21", 1),
#     (-0.9033151361674958, 119.87820157968686, "09/28/18", 1),
#     (32.7767, -96.797, "04/09/21", 0),
#     (-7.6145, 110.7122, "03/31/21", 0),
# ]

# results = []
# print("\n🌍 Running Forest Loss Data Test\n")

# for lat, lon, date, landslide in test_locations:
#     # Parse the date string to get the year
#     month, day, yr = date.split('/')
#     year = int("20" + yr)
    
#     print(f"\n📍 Checking Forest Loss for {lat}, {lon} (Landslide date: {date}, year: {year})...")
#     forest_loss = get_forest_loss(lat, lon, year)
    
#     # Add the result to our list
#     results.append([date, lat, lon, forest_loss, landslide])
    
#     # Print a summary of this test case
#     print(f"Result: Forest Loss = {forest_loss}")

# output_file = os.path.join("dataset", "output.csv")
# os.makedirs(os.path.dirname(output_file), exist_ok=True)

# output_file = "forest_loss_results.csv"
# with open(output_file, mode="w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(["Date", "Latitude", "Longitude", "Forest_Loss", "Landslide"])
#     writer.writerows(results)

# print(f"\n✅ Test Completed. Results saved to {output_file}.")