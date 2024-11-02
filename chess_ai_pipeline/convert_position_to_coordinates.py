import numpy as np
import cv2


def trouver_coordonnees_case(case, output_size):
    # Taille de chaque case de la grille
    square_size = output_size // 8

    # Décoder la colonne et la ligne de la case échiquier (par exemple, 'e4')
    col = ord(case[0].upper()) - 65  # 'A' -> 0, 'B' -> 1, ..., 'H' -> 7
    row = 8 - int(case[1])  # 8 -> 0, 7 -> 1, ..., 1 -> 7

    # Calculer les coordonnées x, y
    x = col * square_size
    y = row * square_size

    # On les met au milieu de la case
    x = x + square_size // 2
    y = y + square_size // 2
    return (x, y)


def appliquer_inverse_transformations(coordonnees, center, angle_rotation, M):
    x, y = coordonnees

    # Étape 1 : Appliquer la rotation inverse
    M_rotation_inv = cv2.getRotationMatrix2D(
        center, -angle_rotation, 1)  # -angle pour l’inversion
    coord_homogeneous = np.array([x, y, 1])  # Coordonnées homogènes
    # Appliquer la rotation inverse
    coord_rotated = np.dot(M_rotation_inv, coord_homogeneous.T)

    # Étape 2 : Appliquer la transformation de perspective inverse
    M_inv = np.linalg.inv(M)
    coords = np.array([[coord_rotated[0], coord_rotated[1]]], dtype="float32")
    coords_perspective_inv = cv2.perspectiveTransform(
        np.array([coords]), M_inv)[0][0]

    return (coords_perspective_inv[0], coords_perspective_inv[1])


def convert_best_moves_to_original_image_coordinates(best_move_white, best_move_black, center, angle_rotation, M_perspective, output_size):
    # Traitement du coup des pièces blanches
    white_case_previous_position = best_move_white[:2]
    white_case_next_position = best_move_white[2:]

    # On doit remmettre ces coordonnées dans le référentiel de l'image originale, OR on a fait deux transformations (homographie + rotation)
    # Etape 1: Annuler la rotation
    coordonnees_white_previous = trouver_coordonnees_case(
        white_case_previous_position, output_size)
    # afficher_position(
    #     image_tournee, [coordonnees_white_previous], label="Coup Blanc")
    coordonnees_white_previous_on_original = appliquer_inverse_transformations(
        coordonnees_white_previous, center, angle_rotation, M_perspective)

    # Etape 2 : Annuler l'homographie
    coordonnees_white_next = trouver_coordonnees_case(
        white_case_next_position, output_size)
    # afficher_position(
    #     image_tournee, [coordonnees_white_next], label="Coup Blanc")
    coordonnees_white_next_on_original = appliquer_inverse_transformations(
        coordonnees_white_next, center, angle_rotation, M_perspective)

    # Traitement du coup des pièces noires
    black_case_previous_position = best_move_black[:2]
    black_case_next_position = best_move_black[2:]

    coordonnees_black_previous = trouver_coordonnees_case(
        black_case_previous_position, output_size)
    # afficher_position(
    #     image_tournee, [coordonnees_black_previous], label="Coup Noir")
    coordonnees_black_previous_on_original = appliquer_inverse_transformations(
        coordonnees_black_previous, center, angle_rotation, M_perspective)

    coordonnees_black_next = trouver_coordonnees_case(
        black_case_next_position, output_size)
    # afficher_position(
    #     image_tournee, [coordonnees_black_next], label="Coup Noir")
    coordonnees_black_next_on_original = appliquer_inverse_transformations(
        coordonnees_black_next, center, angle_rotation, M_perspective)

    return {
        "white": (coordonnees_white_previous_on_original, coordonnees_white_next_on_original),
        "black": (coordonnees_black_previous_on_original, coordonnees_black_next_on_original)
    }
