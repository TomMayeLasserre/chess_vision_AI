# chess_vision_AI

## AI Model Weights

The model weights required to run this project are available for download on Google Drive. Please follow the steps below to set up the weights in the `model_weights` folder.

1. **Download the Weights**:
   - Download the weight files from the following link: [Model Weights Folder](https://drive.google.com/drive/folders/1G2VA3MNB89z0uDn6LtDao64N-vRlIaWH?usp=sharing)

2. **Place the Files**:
   - After downloading, move both `.pt` files into the `model_weights` folder within your project directory.
   - Make sure the files are correctly named according to the model specifications in the code.

By following these steps, you’ll ensure the model weights are set up correctly for use in this project.



## Setting Up Stockfish

This project requires the Stockfish chess engine to evaluate and predict the best moves. Please follow the instructions below to download and set it up.

1. **Download Stockfish**:
   - Go to the official Stockfish website: [Stockfish Download Page](https://stockfishchess.org/download/).
   - Choose the version for **Windows (x86-64)** and download the executable file named `stockfish-windows-x86-64.exe`.

2. **Place the Executable**:
   - After downloading, place the `stockfish-windows-x86-64.exe` file in a folder named `stockfish` within the project directory.
   - The `stockfish_path` in the code should look like this:
     ```python
     stockfish_path = "stockfish/stockfish-windows-x86-64.exe"
     ```
