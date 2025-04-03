
'''
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from PIL import Image
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm

# Initialize MTCNN detector (mtcnn package version)
mtcnn = MTCNN()

# Define paths
input_root = "../original_2"  # Change this to your dataset path
output_root = "cropped_dataset_v2"

# Ensure output directory exists
os.makedirs(output_root, exist_ok=True)

def crop_and_save_faces(input_dir, output_dir):
    """Detects and crops faces from images in input_dir and saves them to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(input_dir), desc=f"Processing {input_dir}"):
        img_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)  # Convert PIL image to NumPy array
            faces = mtcnn.detect_faces(img_array)

            if faces:
                margin = 40  # Increase this value for a bigger crop

                x, y, width, height = faces[0]['box']
                x, y = max(0, x - margin), max(0, y - margin)  # Expand the box
                x2, y2 = x + width + (2 * margin), y + height + (2 * margin)  # Expand bottom-right corner

                cropped_face = img.crop((x, y, x2, y2))  # Crop with margin
                cropped_face = cropped_face.resize((224, 224))  # Resize face

                cropped_face.save(output_path)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

# Loop through directories
for person in os.listdir(input_root):
    input_dir = os.path.join(input_root, person)
    output_dir = os.path.join(output_root, person)
    
    if os.path.exists(input_dir):
        crop_and_save_faces(input_dir, output_dir)

print("Face cropping complete! Cropped images are saved in:", output_root)'''


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from PIL import Image
import numpy as np
from mtcnn import MTCNN

# Initialize MTCNN detector
mtcnn = MTCNN()

def crop_face(image_path, output_path):
    """Detects and crops a face from the given image and saves it."""
    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)  # Convert PIL image to NumPy array
        faces = mtcnn.detect_faces(img_array)

        if faces:
            margin = 40  # Increase this value for a bigger crop

            x, y, width, height = faces[0]['box']
            x, y = max(0, x - margin), max(0, y - margin)  # Expand the box
            x2, y2 = x + width + (2 * margin), y + height + (2 * margin)  # Expand bottom-right corner

            cropped_face = img.crop((x, y, x2, y2))  # Crop with margin
            cropped_face = cropped_face.resize((224, 224))  # Resize face

            cropped_face.save(output_path)
            print(f"Face saved at: {output_path}")
        else:
            print("No face detected.")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Example usage
image_path = "/home/mahmoud/Pictures/me/photo_1_2025-03-28_02-13-31.jpg"  # Change this to your image path
output_path = "cropped_face.jpg"
crop_face(image_path, output_path)
