import requests
import pandas as pd
from osm_handler import OSMHandler
import os
import concurrent.futures
import time
import xml.etree.ElementTree as ET

def get_osm_data(lat, lon, tags, dif=0.007, temp_dir="temp_osm"):
    """
    Get OSM data for a given latitude and longitude.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        tags (list): List of OSM tags to search for
        dif (float): Difference to create bounding box (default: 0.007)
        temp_dir (str): Directory to store temporary OSM files
        
    Returns:
        int: Count of infrastructure elements found
    """
    # Create bounding box
    a1 = lat - dif
    a2 = lat + dif
    b1 = lon - dif
    b2 = lon + dif
    
    # Create unique filename based on coordinates
    f_name = f"{temp_dir}/osm_{lat}_{lon}.osm"
    
    # Get OSM data
    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={b1},{a1},{b2},{a2}"
    
    try:
        r = requests.get(url, timeout=10)
        
        # Check if response is valid XML before proceeding
        if r.status_code != 200:
            print(r.content)
            print(f"Error processing {lat}, {lon}: HTTP status {r.status_code}")
            return 0
            
        # Validate XML content
        try:
            ET.fromstring(r.content)
        except ET.ParseError as xml_err:
            print(f"Error processing {lat}, {lon}: Invalid XML response - {xml_err}")
            return 0
        
        # Save to temporary file
        with open(f_name, 'wb') as f:
            f.write(r.content)
        
        # Process OSM data
        handler = OSMHandler(tags)
        handler.apply_file(f_name)
        count = handler.count()
        
        # Clean up
        if os.path.exists(f_name):
            os.remove(f_name)
        
        return count
    except Exception as e:
        print(f"Error processing {lat}, {lon}: {e}")
        return 0

def process_batch(batch_df, tags, temp_dir):
    """Process a batch of coordinates"""
    results = {}
    for idx, row in batch_df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        count = get_osm_data(lat, lon, tags, temp_dir=temp_dir)
        results[idx] = count
    return results

def add_osm_data_to_csv(input_csv, output_csv, tags_file="tags.txt", batch_size=10, max_workers=5, max_rows=100):
    """
    Add OSM data to the landslide CSV file with parallel processing.
    
    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to output CSV file
        tags_file (str): Path to file containing OSM tags to search for
        batch_size (int): Number of rows to process in each batch
        max_workers (int): Maximum number of parallel workers
        max_rows (int): Maximum number of rows to process
    """
    # Read tags from file
    with open(tags_file, "r") as f:
        tags = f.read().strip().split(", ")
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Limit to max_rows
    df = df.head(max_rows)
    
    # Create temporary directory for OSM files
    temp_dir = "temp_osm"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize results column
    df['osm_count'] = 0
    
    # Process in batches with parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        # Submit batches for processing
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            future = executor.submit(process_batch, batch_df, tags, temp_dir)
            futures[future] = i
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            start_idx = futures[future]
            results = future.result()
            
            # Update the dataframe with results
            for idx, count in results.items():
                df.at[idx, 'osm_count'] = count
            
            print(f"Processed batch starting at index {start_idx}")
    
    # Clean up temporary directory
    try:
        os.rmdir(temp_dir)
    except:
        print(f"Could not remove {temp_dir}, it may not be empty")
    
    # Save the updated CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

if __name__ == "__main__":
    # Example usage
    input_csv = "../../Global Landslide Catalog.csv"
    output_csv = "../../landslide_with_osm.csv"
    add_osm_data_to_csv(input_csv, output_csv)