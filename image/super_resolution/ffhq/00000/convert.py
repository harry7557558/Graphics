import os
from PIL import Image
import random

# Input and Output folders
input_folder = "1024"
output_folder_l = "256"
output_folder_s = "64"

# Create output folders if not exist
os.makedirs(output_folder_l, exist_ok=True)
os.makedirs(output_folder_s, exist_ok=True)

# Function to resize and save as PNG
def resize_and_save_as_png(input_path, output_path, size):
    with Image.open(input_path) as img:
        resized_img = img.resize((size, size), Image.Resampling.BICUBIC)
        resized_img.save(output_path, format="PNG")

# Function to save as JPG with random quality and progressive/non-progressive
def save_as_jpg_with_random_quality(input_path, output_path, size):
    with Image.open(input_path) as img:
        resized_img = img.resize((size, size), Image.Resampling.BICUBIC)
        
        quality = random.randint(45, 95)
        progressive = random.choice([True, False])

        resized_img.save(output_path, format="JPEG", quality=quality, progressive=progressive)

# Loop through each file in the input folder
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".png") and os.path.isfile(os.path.join(input_folder, filename)):
        print(filename)
        # Resize and save as 256x256 PNG
        input_path = os.path.join(input_folder, filename)
        output_path_l = os.path.join(output_folder_l, filename)
        resize_and_save_as_png(input_path, output_path_l, size=256)

        # Save as 128x128 JPG with random quality and progressive/non-progressive
        output_filename_s = os.path.splitext(filename)[0] + ".jpg"
        output_path_s = os.path.join(output_folder_s, output_filename_s)
        save_as_jpg_with_random_quality(input_path, output_path_s, 64)

print("Conversion completed successfully.")
