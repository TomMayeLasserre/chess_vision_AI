import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from stockfish import Stockfish
import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class ChessPosition():
    def __init__(self, stockfish_path="stockfish/stockfish-windows-x86-64.exe", pieces_logo_directory_path='chess_pieces'):
        self.stockfish = Stockfish(path=stockfish_path)
        self.pieces_logo_directory_path = pieces_logo_directory_path

    def parse_fen(self, fen):
        rows = fen.strip().split(' ')[0].split('/')
        board = []
        for row in rows:
            board_row = []
            for char in row:
                if char.isdigit():
                    board_row.extend([' '] * int(char))
                else:
                    board_row.append(char)
            board.append(board_row)
        return board

    def plot_chess_board(self, fen):
        # Analyse de la chaîne FEN
        board = self.parse_fen(fen)

        # Création de la figure et des axes
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        canvas = FigureCanvas(fig)  # Crée un canvas pour récupérer l'image

        # Configuration des limites de l'axe et suppression des axes
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.axis('off')

        # Couleurs des cases
        light_color = '#F0D9B5'  # Couleur des cases claires
        dark_color = '#B58863'   # Couleur des cases sombres

        # Dessin de l'échiquier
        for i in range(8):
            for j in range(8):
                color = light_color if (i + j) % 2 == 0 else dark_color
                square = plt.Rectangle((j, 7 - i), 1, 1, facecolor=color)
                ax.add_patch(square)

        # Correspondance entre les pièces et les images
        piece_images = {
            'P': 'wp.png', 'N': 'wn.png', 'B': 'wb.png', 'R': 'wr.png',
            'Q': 'wq.png', 'K': 'wk.png', 'p': 'bp.png', 'n': 'bn.png',
            'b': 'bb.png', 'r': 'br.png', 'q': 'bq.png', 'k': 'bk.png'
        }

        # Placement des pièces sur l'échiquier
        for i in range(8):
            for j in range(8):
                piece = board[i][j]
                if piece != ' ':
                    img_path = os.path.join(
                        self.pieces_logo_directory_path, piece_images.get(piece))
                    if os.path.exists(img_path):
                        img = mpimg.imread(img_path)
                        imagebox = OffsetImage(img, zoom=0.8)
                        ab = AnnotationBbox(
                            imagebox, (j + 0.5, 7 - i + 0.5), frameon=False)
                        ax.add_artist(ab)

        # Dessiner la figure sur le canvas
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(canvas.get_width_height()[::-1] + (3,))

        # Convertir l'image au format BGR pour OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        plt.close(fig)  # Fermer la figure pour libérer la mémoire

        return img_bgr

    def get_best_moves(self, fen):
        self.stockfish.set_fen_position(fen)
        # Meilleur coup pour les Blancs
        self.stockfish.set_fen_position(fen.replace(
            " b ", " w "))  # Forcer les Blancs à jouer
        best_move_white = self.stockfish.get_best_move()

        # Meilleur coup pour les Noirs
        self.stockfish.set_fen_position(fen.replace(
            " w ", " b "))  # Forcer les Noirs à jouer
        best_move_black = self.stockfish.get_best_move()

        return best_move_white, best_move_black

    def get_evaluation_bars(self, fen):
        """Renvoie les valeurs des barres d'évaluation pour les Blancs et les Noirs."""
        self.stockfish.set_fen_position(fen)
        evaluation = self.stockfish.get_evaluation()

        # Si l'évaluation est en centipions, la valeur sera directement disponible
        if evaluation["type"] == "cp":
            eval_value = evaluation["value"]
        elif evaluation["type"] == "mate":
            # Mat proche, affecte une valeur haute pour illustrer l'avantage
            eval_value = 1000 * evaluation["value"]

        # Normaliser l'évaluation entre -500 et +500 pour un affichage cohérent
        eval_value = max(min(eval_value, 500), -500)

        # Calculer les hauteurs des barres pour chaque côté
        white_bar_height = max(0, eval_value) / 500  # Pour les Blancs
        black_bar_height = max(0, -eval_value) / 500  # Pour les Noirs

        return white_bar_height, black_bar_height
