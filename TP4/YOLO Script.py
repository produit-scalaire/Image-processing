import torch
import cv2
import numpy as np

# Utilisation directe du modèle par téléchargement des poids pré-entrainés via github
# les fichiers seront téléchargés à ~/.cache/torch/hub/checkpoints ou C:\\Users\\<nom_utilisateur>\\.cache\\torch\\hub\\checkpoints
# la liste des modèle est dans https://github.com/pytorch/hub/blob/master/ultralytics_yolov5.md
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


# Une autre solution est de passer les poids d'un modèle pré-entrainé puis de chager le modèle
# Chemin vers le fichier .pt
# model_path = 'yolov5s.pt'  # Remplacez par le chemin de votre modèle .pt
# Charger le modèle PyTorch
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Charger la vidéo
video_path = 'video.mp4'  # Remplacez par le chemin de votre vidéo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur : Impossible de lire la vidéo")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Conversion du frame en RGB (requis pour PyTorch)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Effectuer la détection
    results = model(rgb_frame)

    # Extraire les résultats sous forme de données
    detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, confidence, class
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        label = f"{model.names[int(class_id)]} {confidence:.2f}"
        
        # Dessiner le rectangle de détection
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Afficher la vidéo
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()