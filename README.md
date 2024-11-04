<img width="298" alt="image" src="https://github.com/user-attachments/assets/3abda9cd-c7c8-458f-8d23-beedcc34e0d5">
# chess_vision_ai

**chess_vision_ai** is an AI-powered tool for real-time chess analysis. It leverages computer vision to detect board and pieces from images or video and integrates the 
Stockfish chess engine to recommend moves. Perfect for analyzing games, reviewing plays, and gaining insights in real-time. Et ce depuis n'importe quel angle

image 1 demo : test_data/images/demo.png
image chessboard crée test_data/images/chessboard_demo.png
## Features

- **Board and Piece Detection**: Detects the chessboard and pieces' positions from images or video streams.
- **Move Prediction**: Analyzes board state and recommends the best moves via Stockfish.
- **Real-time Display**: Continuously updates and displays the game state with annotated moves and evaluations.
- **Custom Visualization**: Highlights recommended moves on a virtual chessboard.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/chess_vision_ai.git
   cd chess_vision_ai
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### AI Model Weights

1. **Download Weights**: Download model weights from [Google Drive](https://drive.google.com/drive/folders/1G2VA3MNB89z0uDn6LtDao64N-vRlIaWH?usp=sharing).
2. **Place Files**: Move both `.pt` files to `model_weights` in the project directory with correct names as specified in the code.

### Stockfish Setup

1. **Download Stockfish**: Visit [Stockfish Download Page](https://stockfishchess.org/download/) and download the `stockfish-windows-x86-64.exe` file for Windows.
2. **Place Executable**: Place it in a `stockfish` folder within the project directory, ensuring the `stockfish_path` points to:
   ```python
   stockfish_path = "stockfish/stockfish-windows-x86-64.exe"
   ```

## Usage

Run with a webcam or an image:

- **Using Webcam**:
  ```bash
  python chess_vision_ai.py --source 0
  ```
- **Analyzing an Image**:
  ```bash
  python chess_vision_ai.py --source "path/to/image.jpg"
  ```

### Key Arguments

- `--source`: Path to the image or stream (e.g., `data/images` for an image directory, or `0` for the default webcam).
- `--display_corners`: Display detected board corners on the output image.
- `--display_boxes`: Show bounding boxes around each detected piece.
- `--display_coordinates`: Display coordinates of each detected piece on the board.
- `--display_best_moves`: Show Stockfish's recommended moves on the output.
- `--display_evaluations`: Display move strength evaluations for each side.
- `--view_img`: Show the annotated output in a separate window.

## Examples

```bash
python chess_vision_ai.py --source 0 --display_corners --display_best_moves --stockfish_path "path/to/stockfish"
```

Training
Weights for both models were trained using Colab, jupyter available in jupyter_training



Algorith explenation

Step 1 : extraction of the corners with model 1:
test_data/images/demo_corners.jpg
<img width="298" alt="image" src="https://github.com/user-attachments/assets/412aa5ab-86b7-4eda-9709-339d3ea30260">

Step 2: extraction of the pieces boxes :
test_data/images/demo_boxes.jpg
<img width="305" alt="image" src="https://github.com/user-attachments/assets/91b0a7c0-e779-4ab8-b3c8-0cd9282db9c0">

Step 3: get the coordinates
test_data/images/demo_coordinates.jpg

Step 4: homographie grâce au 4 coins
test_data/images/demo_homographie.jpg
<img width="382" alt="image" src="https://github.com/user-attachments/assets/199e5a67-7d3f-451b-bbce-57d1c4823226">

Step 5 : Trouver le bon sens de l'échequier car on peut prendre l'image depuis n'importe quel angle
Récupèrer les couleurs de chaque cases et les comparer aux couleurs de l'échequier dnas le bon sens avec les blancs en bas + mais il reste deux possibilités possible donc on repère les deux rois et en général le roi blanc eest en bas
test_data/images/demo_reoriente.jpg
<img width="383" alt="image" src="https://github.com/user-attachments/assets/0dce9b5b-80ee-4a36-bf69-c1fb0a7ba5f2">
Step 6 : En déduire les coordonnées des cases et la position FEN

Step 7 : Afficher la position + mettre dans Stockfish pour obtenir meilleur coup 
