from src.embeddings import load_model, load_face_database, match_face
from scripts.get_faces import crop_face

import sys
import os

def main(image_path, model_path, embedding_path):
    if not os.path.isfile(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    input_path = "input_path.jpg"
    crop_face(image_path, input_path)

    model = load_model(model_path)
    database = load_face_database(embedding_path)

    matched_person = match_face(model, database, image_path)
    print(f"The matched person is: {matched_person}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
    else:
        image_path = sys.argv[1]
        model_path = "./models/siamese_model_v1.pth"
        embedding_path = "./embeddings/face_database.pkl"

        main(image_path, model_path, embedding_path)