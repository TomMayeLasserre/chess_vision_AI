
# chess_vision_ai

**chess_vision_ai** is an AI-driven tool designed for real-time chess analysis using computer vision to detect boards and pieces from images or video feeds. The system integrates the Stockfish chess engine to recommend optimal moves, making it ideal for game analysis and insights from any camera angle.

## Demo Images

### Overview Demo:
![Demo Image](test_data/images/demo.png)

### Chessboard Detection:
![Chessboard Demo](test_data/images/chessboard_demo.png)

## Features

- **Board and Piece Detection**: Recognizes chessboard and piece positions from static images or live video.
- **Move Prediction**: Analyzes board positions and suggests optimal moves via Stockfish.
- **Real-time Visualization**: Updates game states with annotated moves and evaluations.
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

1. **Download Weights**: [Download model weights](https://drive.google.com/drive/folders/1G2VA3MNB89z0uDn6LtDao64N-vRlIaWH?usp=sharing).
2. **Place Files**: Save both `.pt` files in the `model_weights` folder, named as specified in the code.

### Stockfish Setup

1. **Download Stockfish**: [Download Stockfish executable](https://stockfishchess.org/download/).
2. **Place Executable**: Save `stockfish-windows-x86-64.exe` in a `stockfish` folder in the project directory and ensure the `stockfish_path` variable points to:
   ```python
   stockfish_path = "stockfish/stockfish-windows-x86-64.exe"
   ```

## Usage

Run the program with a webcam or an image:

- **Using Webcam**:
  ```bash
  python chess_vision_ai.py --source 0
  ```

- **Analyzing an Image**:
  ```bash
  python chess_vision_ai.py --source "path/to/image.jpg"
  ```

### Key Arguments

- `--source`: Path to the image or video stream (`0` for webcam).
- `--display_corners`: Show detected board corners.
- `--display_boxes`: Show bounding boxes around pieces.
- `--display_coordinates`: Display coordinates of each detected piece.
- `--display_best_moves`: Show recommended moves from Stockfish.
- `--display_evaluations`: Show evaluation bars for each side.
- `--view_img`: Display the annotated output in a separate window.

#### Example Usage
```bash
python chess_vision_ai.py --source 0 --display_corners --display_best_moves --stockfish_path "path/to/stockfish"
```

## Training

Weights for the detection models were trained on Google Colab, and the Jupyter notebooks used are available in `jupyter_training`.

## Algorithm Overview

1. **Corner Detection**: Identify the four corners of the chessboard.  
   ![Corners Demo](test_data/images/demo_corners.jpg)

2. **Piece Detection**: Recognize pieces and their bounding boxes.  
   ![Pieces Demo](test_data/images/demo_boxes.jpg)

3. **Calculate Coordinates**: Obtain center coordinates for each detected piece.  
   ![Coordinates Demo](test_data/images/demo_coordinates.jpg)

4. **Homography Transformation**: Transform the board to a top-down view using the detected corners.  
   ![Homography Demo](test_data/images/demo_homographie.jpg)

5. **Board Orientation**: Ensure correct board orientation by matching colors and identifying king locations.  
   ![Orientation Demo](test_data/images/demo_reoriente.jpg)

6. **Square Mapping and FEN Positioning**: Convert piece positions into FEN notation.

7. **Move Analysis**: Display current board position and run it through Stockfish to find the best move.  
   ![Final Demo](test_data/images/demo_final.jpg)

---

This project provides real-time chess analysis from any viewing angle and facilitates insights with advanced visualizations and move recommendations.
