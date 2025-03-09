import requests
import pandas as pd
import datetime
import numpy as np
import rasterio
import os
import sys
import csv

# ✅ Ensure we can import forest data functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "forest")))
from forest_data import get_forest_loss  

# ✅ Read input data
input_data = []
with open("input.txt", "r") as file:
    for line in file:
        lat, lon, date, landslide_flag = line.strip().split()
        lat = float(lat)
        lon = float(lon)
        landslide_flag = int(landslide_flag)
        input_data.append((lat, lon, date, landslide_flag))

# ✅ Convert date format to YYYY-MM-DD
def format_date(date_str):
    return datetime.datetime.strptime(date_str, "%m/%d/%y").strftime("%Y-%m-%d")

def get_weather_data(lat, lon, date):
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    start_date = (datetime.datetime.strptime(date, "%Y-%m-%d") - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "surface_pressure"],
        "timezone": "auto"
    }
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        weather_data = response.json().get("hourly", {})
        print("weather data received")
        return {
            "precipitation": sum(weather_data.get("precipitation", [])) / 16,  # Average over 16 days
            "humidity": sum(weather_data.get("relative_humidity_2m", [])) / 16,
            "wind_speed": sum(weather_data.get("wind_speed_10m", [])) / 16,
            "air_pressure": sum(weather_data.get("surface_pressure", [])) / 16,
            "temperature": sum(weather_data.get("temperature_2m", [])) / 16,
        }
    else:
        print(f"❌ Error fetching weather data for {lat}, {lon}: {response.text}")
        return None

def get_forest_loss_data(lat, lon, date):
    year = int(date[:4])  # Extract the year from the date
    return get_forest_loss(lat, lon, year)

results = []
for lat, lon, date, landslide in input_data:
    formatted_date = format_date(date)
    
    forest_loss = get_forest_loss_data(lat, lon, formatted_date)
    weather = get_weather_data(lat, lon, formatted_date)
    
    if weather:
        results.append([
            formatted_date, lat, lon, forest_loss,
            weather["precipitation"], weather["humidity"],
            weather["wind_speed"], weather["air_pressure"], weather["temperature"],
            landslide
        ])
    else:
        results.append([formatted_date, lat, lon, forest_loss, None, None, None, None, None, landslide])

output_file = os.path.join(os.path.dirname(__file__), "output.csv")
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Date", "Latitude", "Longitude", "Forest_Loss",
        "Precipitation", "Humidity", "Wind_Speed",
        "Air_Pressure", "Temperature", "Landslide"
    ])
    writer.writerows(results)

print(f"\n✅ Data collection complete. Results saved to {output_file}.")
