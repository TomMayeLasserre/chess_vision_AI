import torch


def find_corners_with_yolo(model_weights_path, img_path):
    # Charger le modèle YOLOv5
    model = torch.hub.load(
        'yolov5', 'custom', path=model_weights_path, source='local')
    model.eval()
    results = model([img_path])

    # Extraire les coordonnées des bounding boxes
    # Liste des bounding boxes sous forme [x, y, w, h]
    corner_boxes = results.xyxy[0].cpu().numpy()

    return corner_boxes


def get_corner_coordinates_from_corner_boxes(corner_boxes):
    corners = []
    for box in corner_boxes:

        x_box_size = abs(int(box[2] - box[0]))
        y_box_size = abs(int(box[3]-box[1]))
        x_center = int(box[0]) + x_box_size//2
        y_center = int(box[1]) + y_box_size//2
        corners.append((x_center, y_center))

    # Assurer que les coins sont dans l'ordre (haut-gauche, haut-droit, bas-gauche, bas-droit)
    # Trier d'abord par y (haut-bas), puis par x (gauche-droite)
    corners = sorted(corners, key=lambda p: (p[1], p[0]))

    return corners
