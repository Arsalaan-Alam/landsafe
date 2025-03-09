# import numpy as np
# from PIL import Image
# import os

# Image.MAX_IMAGE_PIXELS = 10**10

# class TifHandler:
#     def __init__(self, file_path):
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"❌ ERROR: File {file_path} not found!")

#         self.file_path = file_path
#         self.ar = np.array(Image.open(self.file_path), dtype=np.uint8)

#         filename = os.path.basename(self.file_path)
        
#         # ✅ Fix: Extract latitude correctly
#         lat_part = filename.split("_")[-2]
#         lat_value = int(lat_part[:-1])  
#         lat_suffix = lat_part[-1]      

#         # ✅ Fix: Extract longitude correctly
#         lon_part = filename.split("_")[-1][:-4]  
#         lon_value = int(lon_part[:-1])  
#         lon_suffix = lon_part[-1]       

#         self.top_left = [lat_value if lat_suffix == "N" else -lat_value,
#                          lon_value if lon_suffix == "E" else -lon_value]

#         self.bottom_right = [self.top_left[0] - 10, self.top_left[1] + 10]
#         self.pixPerMile = int(4000 / 69)

#         print(f"✅ Loaded {filename} | Bounds: {self.top_left} to {self.bottom_right}")

#     def inBounds(self, lat, lon):
#         return (self.bottom_right[0] <= lat <= self.top_left[0]) and (self.top_left[1] <= lon <= self.bottom_right[1])

#     def forestLoss(self, lat, lon):
#         if not self.inBounds(lat, lon):
#             print(f"❌ {lat}, {lon} is out of bounds!")
#             return None, None

#         newLat = self.top_left[0] - lat
#         newLon = lon - self.top_left[1]

#         x = int(newLat * 4000)
#         y = int(newLon * 4000)

#         if 0 <= x < 40000 and 0 <= y < 40000:
#             loss_value = self.ar[x, y]
#             print(f"📊 Raw loss value at ({lat}, {lon}): {loss_value}")

#             return (loss_value > 0), loss_value  
#         else:
#             print(f"❌ Coordinates ({lat}, {lon}) out of valid range!")
#             return None, None
