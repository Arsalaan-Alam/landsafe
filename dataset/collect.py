import requests
import pandas as pd
import datetime
import os
import sys
import csv

# ✅ Ensure we can import forest data functions
from forest.forest_data import get_forest_loss  

# ✅ Import slope data function
from slope_data import getSlope

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

# ✅ Fetch daily weather data from Open-Meteo (last 15 days)
def get_weather_data(lat, lon, date):
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    start_date = (datetime.datetime.strptime(date, "%Y-%m-%d") - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
    end_date = (datetime.datetime.strptime(date, "%Y-%m-%d") - datetime.timedelta(days=7)).strftime("%Y-%m-%d")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["precipitation_sum", "temperature_2m_max", "temperature_2m_min", "surface_pressure_mean", "relative_humidity_2m_max", "relative_humidity_2m_min", "wind_speed_10m_max"],
        "timezone": "auto"
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        weather_data = response.json().get("daily", {})
        print(f"✅ Weather data received for {lat}, {lon}")

        # Extract last 15 to 7 days of data
        return {
            "precipitation": weather_data.get("precipitation_sum", [])[:9],  # Last 15 to 7 days
            "temperature": [(max_ + min_) / 2 for max_, min_ in zip(weather_data.get("temperature_2m_max", []), weather_data.get("temperature_2m_min", []))][:9],
            "air_pressure": weather_data.get("surface_pressure_mean", [])[:9],
            "humidity": [(max_ + min_) / 2 for max_, min_ in zip(weather_data.get("relative_humidity_2m_max", []), weather_data.get("relative_humidity_2m_min", []))][:9],
            "wind_speed": weather_data.get("wind_speed_10m_max", [])[:9]
        }
    else:
        print(f"❌ Error fetching weather data for {lat}, {lon}: {response.text}")
        return None

# ✅ Fetch forest loss data
def get_forest_loss_data(lat, lon, date):
    year = int(date[:4])  # Extract the year from the date
    return get_forest_loss(lat, lon, year)

# ✅ Fetch slope data
def get_slope_data(lat, lon):
    return getSlope(lat, lon)

# ✅ Process input data and store results
results = []
for lat, lon, date, landslide in input_data:
    formatted_date = format_date(date)
    
    forest_loss = get_forest_loss_data(lat, lon, formatted_date)
    weather = get_weather_data(lat, lon, formatted_date)
    slope = get_slope_data(lat, lon)
    
    if weather:
        row = [formatted_date, lat, lon]
        
        # Add climate features for 15 to 7 days before the incident
        for i in range(9):
            row.extend([
                weather["precipitation"][i] if len(weather["precipitation"]) > i else None,
                weather["temperature"][i] if len(weather["temperature"]) > i else None,
                weather["air_pressure"][i] if len(weather["air_pressure"]) > i else None,
                weather["humidity"][i] if len(weather["humidity"]) > i else None,
                weather["wind_speed"][i] if len(weather["wind_speed"]) > i else None,
            ])

        row.append(forest_loss)
        row.append(slope)
        row.append(landslide)
        results.append(row)

# ✅ Write results to output.csv
output_file = os.path.join(os.path.dirname(__file__), "output.csv")
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    
    # Write headers
    headers = ["Date", "Latitude", "Longitude"]
    for i in range(15, 6, -1):
        headers.extend([f"precip{i}", f"temp{i}", f"air{i}", f"humidity{i}", f"wind{i}"])
    headers.extend(["Forest_Loss", "Slope", "Landslide"])
    
    writer.writerow(headers)
    writer.writerows(results)

print(f"\n✅ Data collection complete. Results saved to {output_file}.")
