import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_pieces_boxes(model_path, img_path):
    # Charger le modèle YOLOv5
    model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')
    model.eval()

    # Charger l'image
    img = cv2.imread(img_path)
    results = model([img_path])

    # Extraire les coordonnées des bounding boxes
    # Liste des bounding boxes sous forme [x1, y1, x2, y2, confidence, class]
    boxes = results.xyxy[0].cpu().numpy()
    return boxes
