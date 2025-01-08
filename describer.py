import os
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the BLIP model and processor (an open-source image captioning model)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Define directories
photos_dir = "photos"
described_dir = "described"
os.makedirs(described_dir, exist_ok=True)

# Initialize the CSV data
csv_data = []

# Process images
file_counter = 1
for filename in os.listdir(photos_dir):
    file_path = os.path.join(photos_dir, filename)
    if not filename.lower().endswith(("png", "jpg", "jpeg")):
        continue  # Skip non-image files

    try:
        # Open the image
        image = Image.open(file_path).convert("RGB")
        
        # Generate description
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs)
        description = processor.decode(output[0], skip_special_tokens=True)
        
        # Rename and save the image
        new_name = f"{file_counter:05}.jpg"
        new_path = os.path.join(described_dir, new_name)
        image.save(new_path)
        
        # Append data to CSV list
        csv_data.append({"filename": new_name, "description": description})
        file_counter += 1

        # Delete file
        os.remove(file_path)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Save to CSV
csv_file = os.path.join(described_dir, "index.csv")
pd.DataFrame(csv_data).to_csv(csv_file, index=False)

print(f"Processing complete. Descriptions saved to {csv_file}.")