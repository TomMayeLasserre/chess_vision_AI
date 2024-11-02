import numpy as np
import cv2


def redresser_image(img, corners, pieces_coordinates, output_size):
    # Vérifiez que nous avons exactement quatre points
    if len(corners) == 4:
        # Organiser les points pour perspective transform
        corners = np.array(
            [corners[0], corners[1], corners[2], corners[3]], dtype="float32")

        # Points de destination pour redresser en carré (
        # x output_size)
        dst_pts = np.array([[0, 0], [output_size, 0], [output_size, output_size], [
                           0, output_size]], dtype="float32")

        # Calculer la transformation de perspective
        M = cv2.getPerspectiveTransform(corners, dst_pts)
        image_redressee = cv2.warpPerspective(
            img, M, (output_size, output_size))

        # Transformer les coordonnées des pièces
        pieces_coordinates_format = np.array(
            [[(piece[0], piece[1]) for piece in pieces_coordinates]], dtype="float32")
        new_pieces_coordinates = cv2.perspectiveTransform(pieces_coordinates_format, M)[
            0]  # Appliquez la transformation

        # On remet la proba et le nom de la pièce
        new_pieces_coordinates = [[p[0], p[1], pieces_coordinates[i][2],
                                   pieces_coordinates[i][3]] for i, p in enumerate(new_pieces_coordinates)]

        # Retourner l'image redressée et les nouvelles coordonnées
        return image_redressee, new_pieces_coordinates, M
    else:
        print("Erreur : Nombre de coins détectés différent de 4. Veuillez vérifier les résultats du modèle.")
        return None, None


# Partie 2 : Déterminer le bon sens de l'échequier parmis les 4

def divide_image_into_grid(image):
    height, width = image.shape[:2]
    square_height = height // 8
    square_width = width // 8

    grid = []
    for row in range(8):
        row_squares = []
        for col in range(8):
            x_start = int(col * square_width)
            y_start = int(row * square_height)
            x_end = int(x_start + square_width)
            y_end = int(y_start + square_height)

            square_image = image[y_start:y_end, x_start:x_end]
            row_squares.append(square_image)
        grid.append(row_squares)
    return grid


def determine_square_colors(grid):
    colors = np.zeros((8, 8), dtype=int)
    for row_index, row in enumerate(grid):
        for col_index, square_image in enumerate(row):
            gray_square = cv2.cvtColor(square_image, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray_square)
            color = 1 if mean_intensity > 127 else 0  # 1 pour blanc, 0 pour noir
            colors[row_index, col_index] = color
    return colors


def generate_rotations_of_detected_pattern(detected_pattern):
    rotation_0 = detected_pattern
    rotation_90 = np.rot90(detected_pattern, k=1)
    rotation_180 = np.rot90(detected_pattern, k=2)
    rotation_270 = np.rot90(detected_pattern, k=3)

    rotations = {
        '0': rotation_0,
        '90': rotation_90,
        '180': rotation_180,
        '270': rotation_270
    }
    return rotations


def compare_rotated_patterns_to_desired(rotated_patterns):

    # Etape 1 : Trouver de combien tourner pour que ça matche avec blanc en bas et noir en haut
    desired_pattern = [[1, 0, 1, 0, 1, 0, 1, 0],
                       [0, 1, 0, 1, 0, 1, 0, 1],
                       [1, 0, 1, 0, 1, 0, 1, 0],
                       [0, 1, 0, 1, 0, 1, 0, 1],
                       [1, 0, 1, 0, 1, 0, 1, 0],
                       [0, 1, 0, 1, 0, 1, 0, 1],
                       [1, 0, 1, 0, 1, 0, 1, 0],
                       [0, 1, 0, 1, 0, 1, 0, 1]]

    best_match = None
    min_difference = float('inf')

    for rotation_angle, rotated_pattern in rotated_patterns.items():
        difference = np.sum(rotated_pattern != desired_pattern)
        if difference < min_difference:
            min_difference = difference
            best_match = rotation_angle

    return best_match


def adjust_angle_for_rotation(best_rotation):
    angle_map = {
        '0': 0,
        '90': 90,
        '180': 180,
        '270': 270
    }
    return angle_map[best_rotation]


def get_rotation_matrix(image_shape, angle):
    (h, w) = image_shape[:2]
    center = (w / 2, h / 2)
    angle_float = float(angle)
    # Calculer la matrice de rotation (angle positif pour rotation antihoraire)
    M = cv2.getRotationMatrix2D(center, angle_float, 1.0)
    return M, center


def rotate_piece_positions(pieces_coordinates, M):
    # Convertir la liste des coordonnées en array
    points = np.array([[coord[0], coord[1]]
                      for coord in pieces_coordinates], dtype='float32')
    # Ajouter une dimension pour cv2.transform
    points = points.reshape(-1, 1, 2)
    # Appliquer la transformation
    rotated_points = cv2.transform(points, M)
    rotated_pieces_coordinates = []
    for i, coord in enumerate(pieces_coordinates):
        rx, ry = rotated_points[i][0][0], rotated_points[i][0][1]
        rotated_pieces_coordinates.append([rx, ry, coord[2], coord[3]])
    return rotated_pieces_coordinates


def mettre_image_dans_le_bon_sens(image, pieces_coordinates, output_size):
    # Étape 1 : Diviser l'image en grille
    grid = divide_image_into_grid(image)

    # Étape 2 : Déterminer la couleur de chaque case
    detected_pattern = determine_square_colors(grid)

    # Étape 3 : Générer les rotations du motif détecté
    rotated_patterns = generate_rotations_of_detected_pattern(detected_pattern)
    # Étape 4 : Comparer les rotations du motif détecté avec le motif standard souhaité
    best_rotation = compare_rotated_patterns_to_desired(rotated_patterns)

    # On fait les rotations après ça (de 0 ou 90°)
    # Étape 5 : Calculer la matrice de rotation M
    angle = adjust_angle_for_rotation(best_rotation)
    M_rotation, center = get_rotation_matrix(image.shape, angle)

    # Étape 6 : Appliquer la rotation à l'image
    image_tournee = cv2.warpAffine(
        image, M_rotation, (image.shape[1], image.shape[0]))

    # Étape 7 : Appliquer la rotation aux positions des pièces
    new_pieces_coordinates = rotate_piece_positions(
        pieces_coordinates, M_rotation)
    # Partie 2 : déterminer l'orientation exacte de l'échiquier en utilisant uniquement les couleurs des cases des coins ne suffit pas, car cela laisse une ambiguïté entre deux orientations (une rotation de 180 degrés). Pour résoudre ce problème et déterminer l'orientation complète de l'échiquier, il est nécessaire d'utiliser des informations supplémentaires.
    # On suppose alors que l'échequier est au moins dans le bon sens vertical !
    # Donc maintenant on regarde la position de rois et on voit si le noir est en bas ou en haut

    # Roi noir : index 2
    # Roi blanc : index 8

    y_black_king = 0
    y_white_king = 0
    for piece in new_pieces_coordinates:
        if piece[3] == 2:
            y_black_king = piece[1]
        elif piece[3] == 8:
            y_white_king = piece[1]

    # On regarde dans quel moitié les rois sont : (on regarde que selon y)
    if y_black_king < output_size//2:  # Le roi noir est en haut
        # Donc on a rien besoin de faire
        other_angle = 0
    else:  # Le roi noir est en bas, ce qui n'est pas trop normal
        # Donc on fait une rotation de -180°
        other_angle = -180

    # On refait les rotations :
    M_rotation2, center = get_rotation_matrix(image.shape, other_angle)

    # Étape 6 : Appliquer la rotation à l'image
    image_tournee_v2 = cv2.warpAffine(
        image_tournee, M_rotation2, (image_tournee.shape[1], image_tournee.shape[0]))

    # Étape 7 : Appliquer la rotation aux positions des pièces
    new_pieces_coordinates_v2 = rotate_piece_positions(
        new_pieces_coordinates, M_rotation2)

    angle_ratation_final = angle + other_angle
    return image_tournee_v2, new_pieces_coordinates_v2, center, angle_ratation_final
