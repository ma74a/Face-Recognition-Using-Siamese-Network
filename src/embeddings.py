import torch

from scipy.spatial.distance import euclidean
from PIL import Image
import pickle
import os


from scripts.get_faces import crop_face
from src.model import SiameseNetwork
from utils.config import * 






def load_model(path="../models/siamese_model_v1.pth"):
        model = SiameseNetwork()
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        print(f"Model loaded from {path}")
        model.eval()
        return model



def extract_embeddings(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = TRANSFORM["test"](img).unsqueeze(0)

    with torch.no_grad():
        out = model.forward_once(img).squeeze()
    
    return out.cpu().numpy()



def face_embeddings_dataset(model, data_dir):
    database = {}

    for cls_name in os.listdir(data_dir):
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        images_emd = []
        for img_name in os.listdir(cls_dir):
            if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(cls_dir, img_name)
            try:
                img_emb = extract_embeddings(model, img_path)
                images_emd.append(img_emb)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    

        
        database[cls_name] = images_emd

    return database

def match_face(model, database, img_path, distance_threshold=0.9):
    query_embedding = extract_embeddings(model, img_path)
    matches = []
    
    for class_name, embeddings in database.items():
        if not embeddings:
            continue
        
        # Calculate distances for all embeddings of this class
        distances = [euclidean(query_embedding, emb) for emb in embeddings]
        
        # Get the lowest distance for this class
        best_distance = min(distances)
        
        if best_distance <= distance_threshold:
            matches.append((class_name, best_distance))
    
    # Sort matches by distance (lowest first)
    matches.sort(key=lambda x: x[1])
    
    # Return top k matches
    return matches[:1]



def save_face_database(database, filepath="../embeddings/face_database.pkl"):
    """Save the face database to disk using pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(database, f)
    print(f"Database saved to {filepath}")



def load_face_database(filepath="../embeddings/face_database.pkl"):
    """Load the face database from disk"""
    with open(filepath, 'rb') as f:
        database = pickle.load(f)
    print(f"Database loaded from {filepath} with {len(database)} identities")
    return database


if __name__ == "__main__":
    model_path = "../models/siamese_model_v1.pth"
    data_dir = "/home/mahmoud/try1/dataset/train"

    embedding_path = "../embeddings/face_database.pkl"

    model = load_model(model_path)

    # database = face_embeddings_dataset(model, data_dir)
    # save_face_database(database, embedding_path)
    database = load_face_database(embedding_path)

    image_path = "/home/mahmoud/Pictures/meeee.jpg" 
    output_path = "input_face.jpg"
    output_path = crop_face(image_path, output_path)

    # print(match_face2(model, database, output_path))
    matches = match_face(model, database, output_path)
    if matches:
        for name, dist in matches:
            print(f"Matched with {name} (Distance: {dist:.4f})")
    else:
        print("No match found.")
