import cv2
import matplotlib.pyplot as plt


def trouver_cases_pieces(new_pieces_coordinates, output_size):
    # Taille de chaque case de la grille
    square_size = output_size // 8  # Taille en pixels de chaque case

    # Dictionnaire pour stocker la case de chaque pièce
    pieces_positions = []

    # Identifier les cases pour chaque coordonnée
    for coord in new_pieces_coordinates:
        x, y = int(coord[0]), int(coord[1])

        # Calculer la ligne et la colonne de la case où se trouve la pièce
        col = x // square_size  # Colonne de la case (de 0 à 7)
        row = y // square_size  # Ligne de la case (de 0 à 7)

        # Convertir en notation échiquier (par exemple, 'A1', 'B2', etc.)
        case = f"{chr(65 + col)}{8 - row}"
        # Stocke la case et les coordonnées
        pieces_positions.append((case, (x, y), coord[2], coord[3]))

    return pieces_positions  # Retourner les positions des pièces


def coordinates_to_fen(new_pieces_coordinates):
    # Initialisation de l'échiquier vide
    board = [['1' for _ in range(8)] for _ in range(8)]
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    # Mapping des index de pièces aux notations FEN
    index_to_fen = {
        0.0: 'B',  # 'bishop' ambigu, on le considère comme un fou blanc
        1.0: 'b',  # 'black-bishop'
        2.0: 'k',  # 'black-king'
        3.0: 'n',  # 'black-knight'
        4.0: 'p',  # 'black-pawn'
        5.0: 'q',  # 'black-queen'
        6.0: 'r',  # 'black-rook'
        7.0: 'B',  # 'white-bishop'
        8.0: 'K',  # 'white-king'
        9.0: 'N',  # 'white-knight'
        10.0: 'P',  # 'white-pawn'
        11.0: 'Q',  # 'white-queen'
        12.0: 'R',  # 'white-rook'
    }

    for piece in new_pieces_coordinates:
        square = piece[0]
        index = piece[-1]  # Dernier élément
        fen_letter = index_to_fen.get(index)
        if fen_letter is None:
            continue  # Ignore si l'index n'est pas dans le mapping
        file_char = square[0].lower()
        rank_char = square[1]
        if file_char not in files or not rank_char.isdigit():
            continue  # Ignore les cases invalides
        file_index = files.index(file_char)
        rank_index = 8 - int(rank_char)  # Rangs de 8 à 1
        if 0 <= rank_index < 8 and 0 <= file_index < 8:
            board[rank_index][file_index] = fen_letter

    # Génération de la chaîne FEN pour les pièces
    fen_rows = []
    for row in board:
        fen_row = ''
        empty_count = 0
        for square in row:
            if square == '1':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += square
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    fen_piece_placement = '/'.join(fen_rows)

    # Assemblage de la FEN complète
    fen = f"{fen_piece_placement} w - - 0 1"
    return fen
