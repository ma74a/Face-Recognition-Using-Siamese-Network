import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import flet as ft
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import os
from src.embeddings import load_model, load_face_database, crop_face, match_face
import tempfile

class FaceRecognitionApp:
    def __init__(self):
        self.cap = None
        self.is_capturing = False
        self.model = None
        self.database = None
        self.last_recognized = None
        self.last_recognition_time = None
        self.attendance_file = "attendance.csv"
        
        # Initialize attendance file if it doesn't exist
        if not os.path.exists(self.attendance_file):
            pd.DataFrame(columns=['Name', 'Time']).to_csv(self.attendance_file, index=False)
        
        # Load model and database
        self.model = load_model()
        self.database = load_face_database()

    def initialize_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")

    def release_camera(self):
        if self.cap is not None:
            self.cap.release()

    def process_frame(self, frame):
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Save frame temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
            pil_image.save(temp_path)
        
        try:
            # Crop face
            cropped_path = crop_face(temp_path, temp_path)
            
            # Match face
            matches = match_face(self.model, self.database, cropped_path)
            
            if matches:
                name, distance = matches[0]
                current_time = datetime.now()
                
                # Check if this is a new recognition or enough time has passed
                if (self.last_recognized != name or 
                    (current_time - self.last_recognition_time).total_seconds() > 5):
                    
                    # Update attendance
                    df = pd.read_csv(self.attendance_file)
                    df = pd.concat([df, pd.DataFrame({
                        'Name': [name],
                        'Time': [current_time.strftime('%Y-%m-%d %H:%M:%S')]
                    })], ignore_index=True)
                    df.to_csv(self.attendance_file, index=False)
                    
                    self.last_recognized = name
                    self.last_recognition_time = current_time
                    return name, distance
            
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
        
        return None, None

    def main(self, page: ft.Page):
        page.title = "Face Recognition Attendance System"
        page.theme_mode = ft.ThemeMode.DARK
        page.padding = 20
        
        # Create UI elements
        self.image = ft.Image(
            src=None,
            width=640,
            height=480,
            fit=ft.ImageFit.CONTAIN,
        )
        
        self.status_text = ft.Text(
            value="Status: Ready",
            size=20,
            weight=ft.FontWeight.BOLD,
        )
        
        self.recognized_name = ft.Text(
            value="",
            size=24,
            weight=ft.FontWeight.BOLD,
            color=ft.colors.GREEN,
        )
        
        # Create buttons
        self.start_button = ft.ElevatedButton(
            text="Start Camera",
            on_click=self.start_camera,
        )
        
        self.stop_button = ft.ElevatedButton(
            text="Stop Camera",
            on_click=self.stop_camera,
            disabled=True,
        )
        
        # Layout
        page.add(
            ft.Column([
                ft.Row([
                    self.start_button,
                    self.stop_button,
                ], alignment=ft.MainAxisAlignment.CENTER),
                self.image,
                self.status_text,
                self.recognized_name,
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        )

    def start_camera(self, e):
        try:
            self.initialize_camera()
            self.is_capturing = True
            self.start_button.disabled = True
            self.stop_button.disabled = False
            self.status_text.value = "Status: Camera Active"
            self.update_camera()
        except Exception as e:
            self.status_text.value = f"Error: {str(e)}"

    def stop_camera(self, e):
        self.is_capturing = False
        self.release_camera()
        self.start_button.disabled = False
        self.stop_button.disabled = True
        self.status_text.value = "Status: Camera Stopped"
        self.image.src = None
        self.recognized_name.value = ""
        self.page.update()

    def update_camera(self):
        if not self.is_capturing:
            return
            
        ret, frame = self.cap.read()
        if ret:
            # Process frame for face recognition
            name, distance = self.process_frame(frame)
            
            if name:
                self.recognized_name.value = f"Recognized: {name}"
                self.recognized_name.color = ft.colors.GREEN
            else:
                self.recognized_name.value = "No face recognized"
                self.recognized_name.color = ft.colors.RED
            
            # Convert frame to base64 for display
            _, buffer = cv2.imencode('.jpg', frame)
            self.image.src = f"data:image/jpeg;base64,{buffer.tobytes().hex()}"
            self.page.update()
        
        # Schedule next update
        self.page.after(100, self.update_camera)

if __name__ == "__main__":
    app = FaceRecognitionApp()
    ft.app(target=app.main) 