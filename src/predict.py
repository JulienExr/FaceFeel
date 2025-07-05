import cv2
import torch
import torchvision.transforms as transforms
from model import EmotionRecognitionModel
from dataloader import get_dataloader
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

def load_model(path, device):
    model = EmotionRecognitionModel(num_classes=8)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_face(face, transform=None):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5
    face = np.expand_dims(face, axis=0)
    face = torch.tensor(face, dtype=torch.float32).unsqueeze(0)
    return face


def predict_image(model, face_tensor, device, class_names):
    face_tensor = face_tensor.to(device)
    with torch.no_grad():
        output = model(face_tensor)
        _, predicted = torch.max(output, 1)
        predicted_idx = predicted.item()
        
        if predicted_idx >= len(class_names):
            return f"Unknown_Class_{predicted_idx}"
        
        return class_names[predicted_idx]

def predict_from_webcam(model, device, class_names, transform):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ Webcam not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            cv2.putText(frame, "No face", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        else:
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                try:
                    face_tensor = preprocess_face(face_img, transform)
                    emotion = predict_image(model, face_tensor, device, class_names)
                    cv2.putText(frame, emotion, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                except Exception as e:
                    print("❌ Erreur de prédiction :", e)
                    cv2.putText(frame, "Error", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Facefeel - Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def predict_from_image(image_path, model, device, class_names, transform):
    img = cv2.imread(image_path)
    face_tensor = preprocess_face(img, transform)
    emotion = predict_image(model, face_tensor, device, class_names)
    cv2.putText(img, emotion, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image", "webcam"], default="image")
    parser.add_argument("--image", type=str, help="Chemin vers l'image si mode image")
    parser.add_argument("--model", type=str, default=os.path.join(PROJECT_ROOT, "experiments", "checkpoints", "best_model.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])

    class_names = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

    model = load_model(args.model, device)

    if args.mode == "webcam":
        predict_from_webcam(model, device, class_names, transform)
    else:
        if not args.image:
            print("❌ Merci de spécifier un chemin d'image avec --image")
        else:
            predict_from_image(args.image, model, device, class_names, transform)
