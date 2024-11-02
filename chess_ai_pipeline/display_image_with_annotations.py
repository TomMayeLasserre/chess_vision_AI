import cv2
import matplotlib.pyplot as plt


def place_best_move_on_image(image, best_move_white_coordinates, best_move_black_coordinates, white_color=(255, 0, 0), black_color=(0, 0, 255)):  # Bleu et rouge
    # Fonction pour dessiner le déplacement sur l'image
    def dessiner_deplacement(image, coordonnees_previous_position_on_original_image, coordonnees_next_position_on_original_image, color):

        # Dessiner la position initiale (cercle)
        cv2.circle(image,
                   (int(coordonnees_previous_position_on_original_image[0]), int(
                       coordonnees_previous_position_on_original_image[1])),
                   radius=5,
                   color=color,  # Couleur spécifique pour chaque pièce
                   thickness=-1)

        # Dessiner la position finale (cercle)
        cv2.circle(image,
                   (int(coordonnees_next_position_on_original_image[0]), int(
                       coordonnees_next_position_on_original_image[1])),
                   radius=5,
                   color=color,
                   thickness=-1)

        # Dessiner une flèche de la position initiale vers la position finale
        cv2.arrowedLine(image,
                        (int(coordonnees_previous_position_on_original_image[0]), int(
                            coordonnees_previous_position_on_original_image[1])),
                        (int(coordonnees_next_position_on_original_image[0]), int(
                            coordonnees_next_position_on_original_image[1])),
                        color=color,
                        thickness=3,
                        tipLength=0.2)  # Taille de la pointe de la flèche
    # Traiter le coup des pièces blanches avec la couleur bleue
    dessiner_deplacement(
        image, best_move_white_coordinates[0], best_move_white_coordinates[1], color=white_color)

    # Traiter le coup des pièces noires avec la couleur rouge
    dessiner_deplacement(
        image, best_move_black_coordinates[0], best_move_black_coordinates[1], color=black_color)  # Rouge

    # Retourner l'image avec les annotations
    return image


# Fonction pour afficher les coins détectés
def display_corners_on_image(img, corners, color=(255, 255, 0)):  # Jaune
    img_copy = img.copy()
    for c in corners:
        cv2.circle(img_copy, (int(c[0]), int(c[1])), 5, color, -1)
    return img_copy

# Fonction pour afficher les bounding boxes


def display_boxes_on_image(img, boxes, color=(255, 0, 0)):  # Bleu clair
    img_copy = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
    return img_copy

# Fonction pour afficher les coordonnées des pièces


# Mauve
def display_pieces_coordinates_on_image(img, pieces_coordinates, color=(255, 0, 255)):
    img_copy = img.copy()
    for coord in pieces_coordinates:
        x, y = int(coord[0]), int(coord[1])
        cv2.circle(img_copy, (x, y), 5, color, -1)
    return img_copy

# Fonction pour afficher l'image redressée avec les nouvelles coordonnées


# Vert
def display_image_redressee_and_pieces_positions(image_redressee, new_pieces_coordinates, color=(0, 255, 0)):
    img_copy = image_redressee.copy()
    for coord in new_pieces_coordinates:
        x, y = int(coord[0]), int(coord[1])
        cv2.circle(img_copy, (x, y), 5, color, -1)
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.title("Image redressée avec coordonnées des pièces")
    plt.axis('off')
    plt.show()


def display_barres_evaluation(img, white_bar_height, black_bar_height, white_color, black_color):
    # Dimensions de l'image
    img_height, img_width = img.shape[:2]

    # Paramètres de la barre (largeur et hauteur)
    bar_width = 30
    bar_height = img_height  # Hauteur de la barre couvrant toute la hauteur de l'image

    # Positions pour les barres (aucun espacement avec les bords)
    bar_position_white_x = 0  # Bord gauche de l'image
    bar_position_black_x = img_width - bar_width  # Bord droit de l'image

    # Définition de la zone de la barre (du haut vers le bas)
    y_top = 0
    y_bottom = img_height

    # Dessiner le fond de la barre pour les Blancs (couleur grise)
    cv2.rectangle(img, (bar_position_white_x, y_top),
                  (bar_position_white_x + bar_width, y_bottom), (0, 0, 0), -1)

    # Calcul de la hauteur en pixels pour la barre des Blancs
    white_bar_pixels = int(white_bar_height * bar_height)

    # Dessiner la barre des Blancs (blanc) depuis le bas vers le haut, seulement si white_bar_height > 0
    if white_bar_height > 0:
        cv2.rectangle(img, (bar_position_white_x, y_bottom - white_bar_pixels),
                      (bar_position_white_x + bar_width, y_bottom), (255, 255, 255), -1)

    # Dessiner le contour de la barre des Blancs
    cv2.rectangle(img, (bar_position_white_x, y_top),
                  (bar_position_white_x + bar_width, y_bottom), white_color, 2)

    # Dessiner le fond de la barre pour les Noirs (couleur grise)
    cv2.rectangle(img, (bar_position_black_x, y_top),
                  (bar_position_black_x + bar_width, y_bottom), (255, 255, 255), -1)

    # Calcul de la hauteur en pixels pour la barre des Noirs
    black_bar_pixels = int(black_bar_height * bar_height)

    # Dessiner la barre des Noirs (noir) depuis le bas vers le haut, seulement si black_bar_height > 0
    if black_bar_height > 0:
        cv2.rectangle(img, (bar_position_black_x, y_bottom - black_bar_pixels),
                      (bar_position_black_x + bar_width, y_bottom), (0, 0, 0), -1)

    # Dessiner le contour de la barre des Noirs
    cv2.rectangle(img, (bar_position_black_x, y_top),
                  (bar_position_black_x + bar_width, y_bottom), black_color, 2)

    return img


# Fonction principale pour afficher tout sur l'image originale


def display_everything_on_original_image(img, corners, boxes, pieces_coordinates, coordonnees_best_moves,
                                         white_bar_height, black_bar_height, display_corners=True, display_boxes=True,
                                         display_coordinates=True, display_best_moves=True, display_evaluations=True, white_color=(255, 0, 0), black_color=(0, 0, 255)):  # Bleu et rouge
    img_display = img.copy()

    if display_corners:
        try:
            img_display = display_corners_on_image(img_display, corners)
        except Exception:
            pass
    if display_boxes:
        try:
            img_display = display_boxes_on_image(img_display, boxes)
        except Exception:
            pass
    if display_coordinates:
        try:
            img_display = display_pieces_coordinates_on_image(
                img_display, pieces_coordinates)
        except Exception:
            pass
    if display_best_moves:
        try:
            img_display = place_best_move_on_image(
                img_display, coordonnees_best_moves["white"], coordonnees_best_moves["black"], white_color, black_color)
        except Exception:
            pass
    if display_evaluations:
        try:
            img_display = display_barres_evaluation(
                img_display, white_bar_height, black_bar_height, white_color, black_color)
        except Exception:
            pass

    return img_display
