import torchvision.transforms as transforms
from PIL import Image
import os

# Define augmentations
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=10, scale=(0.9, 1.1), shear=10),
])

# Function to apply augmentations and save images
def augment_and_save(image_path, save_dir, num_augmented=5):
    image = Image.open(image_path).convert('RGB')

    # Save original image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image.save(os.path.join(save_dir, f"{base_name}_original.jpg"))

    # Generate and save augmented images
    for i in range(num_augmented):
        augmented_image = augmentations(image)
        augmented_image.save(os.path.join(save_dir, f"{base_name}_aug_{i}.jpg"))

# Apply augmentation to all images in a dataset folder
def augment_dataset(dataset_dir, save_dir, num_augmented=5):
    os.makedirs(save_dir, exist_ok=True)
    for person in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person)
        save_dir_person = os.path.join(save_dir, person)
        os.makedirs(save_dir_person, exist_ok=True)
        for filename in os.listdir(person_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                augment_and_save(os.path.join(person_path, filename), save_dir_person, num_augmented)

            

# Example usage
augment_dataset("./cropped_dataset_v2", "augmented_faces", num_augmented=5)
