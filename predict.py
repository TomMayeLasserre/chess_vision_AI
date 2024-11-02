import pathlib
import argparse
import os
# new_path = 'yolov5'
# os.chdir(new_path)

import platform
import sys

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    increment_path,
    non_max_suppression,
    scale_boxes,
    print_args
)
from utils.torch_utils import select_device, smart_inference_mode

from chess_ai_pipeline.display_position_and_predict_moves import *
from chess_ai_pipeline.corner_detection import *
from chess_ai_pipeline.pieces_detection import *
from chess_ai_pipeline.perspective_correction import *
from chess_ai_pipeline.chess_position_utils import *
from chess_ai_pipeline.convert_position_to_coordinates import *
from chess_ai_pipeline.display_image_with_annotations import *

from chess_ai_pipeline.model_functions import *


def compute_model_outputs_into_best_move(img, corner_boxes, pieces_boxes, ChessBoard, stockfish_path, pieces_logo_directory_path, display_corners=True, display_boxes=False, display_coordinates=False, display_best_moves=True, display_evaluations=True, white_color=(255, 0, 0), black_color=(0, 0, 255)):

    corners = get_corner_coordinates_from_corner_boxes(corner_boxes)
    # Etape 3 : Calculer les coordinates des pièces à partir de leur box
    # (le milieu selon x, et le MEME milieu selon y)(il ne faut pas prendre le vrai milieu de la box car les pièces sont hautes mais pas larges)
    pieces_coordinates = []
    for box in pieces_boxes:
        delta = abs(box[2]-box[0])/2
        x = int(box[0] + delta)
        y = int(abs(box[3] - delta))
        # De la forme [x, y, proba, classe]
        pieces_coordinates.append([x, y, box[4], box[5]])

    # Etape 3 : Homographie
    output_size = 512   # Taille de l'image redressée (après transformation)
    # Redresser l'image et obtenir les nouvelles coordonnées des pièces
    image_redressee, new_pieces_coordinates, M = redresser_image(
        img.copy(), corners, pieces_coordinates, output_size)

    # Partie 4 : Choisir et tourner l'image dans le bon sens parmis les 4 possibilités (deux longues sous étapes)
    image_tournee, new_pieces_coordinates, center, angle = mettre_image_dans_le_bon_sens(
        image_redressee, new_pieces_coordinates, output_size)

    # Etape 5 : Déterminer les cases (h4...) des pièces sur l'échequier dans le bon sens
    pieces_positions = trouver_cases_pieces(
        new_pieces_coordinates, output_size)

    # Etape 6 : Conversion des cases en positions FEN pour être plus par Stockfish
    fen = coordinates_to_fen(pieces_positions)

    # Etape 6 : Visualisation d'un échequier virtuel
    # ChessBoard.plot_chess_board(fen)

    # Etape 7 : Prédiction du meilleur coup par Stockfish
    best_move_white, best_move_black = ChessBoard.get_best_moves(fen)
    white_bar_height, black_bar_height = ChessBoard.get_evaluation_bars(fen)

    # Etape 8 : Convertir cette position h4 en coordonnées sur l'image de base (annuler tous les changements de perspectives et rotation de l'image effectuées)
    coordonnees_best_moves = convert_best_moves_to_original_image_coordinates(
        best_move_white, best_move_black, center, angle, M, output_size)

    # Etape 9 : Afficher les coordonnées de ces meilleurs coups sur l'image originale
    # image_annotated = place_best_move_on_image(cv2.imread(img_path), coordonnees_best_moves["white"], coordonnees_best_moves["black"])
    image_annotated = display_everything_on_original_image(img.copy(), corners, pieces_boxes, pieces_coordinates, coordonnees_best_moves, white_bar_height, black_bar_height,
                                                           display_corners=display_corners, display_boxes=display_boxes, display_coordinates=display_coordinates, display_best_moves=display_best_moves, display_evaluations=display_evaluations, white_color=white_color, black_color=black_color)

    return image_annotated


def filter_close_pieces(pieces_boxes):
    # Vérification pour éviter le calcul si pieces_boxes est vide
    if len(pieces_boxes) == 0:
        return pieces_boxes  # Retourne un tableau vide si aucune pièce n'est détectée

    # Calcul de la distance minimale requise entre deux pièces
    min_distance = min(
        np.mean(pieces_boxes[:, 2] - pieces_boxes[:, 0]
                ) if len(pieces_boxes) > 0 else 0,
        np.mean(pieces_boxes[:, 3] - pieces_boxes[:, 1]
                ) if len(pieces_boxes) > 0 else 0
    )

    # Suivi des indices des boxes à conserver
    keep_indices = np.ones(len(pieces_boxes), dtype=bool)

    # Comparer chaque box avec les autres
    for i in range(len(pieces_boxes)):
        if not keep_indices[i]:  # Si la box a déjà été marquée pour suppression, on continue
            continue

        x1_i, y1_i, x2_i, y2_i, score_i, class_i = pieces_boxes[i]
        center_i = ((x1_i + x2_i) / 2, (y1_i + y2_i) / 2)

        for j in range(i + 1, len(pieces_boxes)):
            # Si la box a déjà été marquée pour suppression, on continue
            if not keep_indices[j]:
                continue

            x1_j, y1_j, x2_j, y2_j, score_j, class_j = pieces_boxes[j]
            center_j = ((x1_j + x2_j) / 2, (y1_j + y2_j) / 2)

            # Calcul de la distance entre les centres des deux boxes
            distance = np.sqrt(
                (center_i[0] - center_j[0]) ** 2 +
                (center_i[1] - center_j[1]) ** 2
            )

            # Vérification de la proximité
            if distance < min_distance:
                # Supprimer la box avec le score le plus faible
                if score_i < score_j:
                    keep_indices[i] = False
                else:
                    keep_indices[j] = False

    # Retourner les boxes filtrées
    filtered_boxes = pieces_boxes[keep_indices]
    return filtered_boxes


@smart_inference_mode()
def run(
    weights1="models_weights/best_yolo_corner_detection.pt",  # model path or triton URL
    weights2="models_weights/best_yolov5x_pieces_detection_v2.pt",
    source="data/images",  # file/dir/URL/glob/screen/0(webcam)
    stockfish_path='stockfish/stockfish-windows-x86-64.exe',
    pieces_logo_directory_path='chess_pieces',
    display_corners=True,
    display_boxes=False,
    display_coordinates=False,
    display_best_moves=True,
    display_evaluations=True,
    white_color=(255, 0, 0), black_color=(0, 0, 255),
    data="data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project="runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):

    # Manip nécessaire car j'ai entrainé les modèles sur Linux et je fais l'inférence sur Windows
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    source = str(source)
    save_img = not nosave and not source.endswith(
        ".txt")  # save inference images

    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(
        ".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Dossiers
    # créer un dossier incrémental
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    print(weights1)
    print(weights2)
    # CHARGEMENT DES DEUX MODELES
    device = select_device(device)
    model_corner = DetectMultiBackend(
        weights1, device=device, dnn=dnn, data=data, fp16=half)
    model_pieces = DetectMultiBackend(
        weights2, device=device, dnn=dnn, data=data, fp16=half)  # second modèle
    stride, names, pt = model_corner.stride, model_corner.names, model_corner.pt
    imgsz = check_img_size(imgsz, s=stride)  # vérifier la taille de l'image

    # Chargement de Stockfish
    ChessBoard = ChessPosition(stockfish_path, pieces_logo_directory_path)

    # Déterminer si la source est un flux webcam (entier) ou une image fixe (chemin)
    # True si source est un chemin (image), False si webcam (entier)
    print(source)

    # True si source est un chemin (image), False si webcam (chaîne numérique)
    is_image_file = not source.isdigit()

    print('Type de source :', is_image_file)
    # Création du dataloader
    bs = 1  # taille de batch
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz,
                              stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(
            source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz,
                             stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Inférence
    # échauffement modèle 1
    model_corner.warmup(
        imgsz=(1 if pt or model_corner.triton else bs, 3, *imgsz))
    # échauffement modèle 2
    model_pieces.warmup(
        imgsz=(1 if pt or model_pieces.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(
        device=device), Profile(device=device))

    # Boucle sur le dataset
    for path, im, im0s, vid_cap, s in dataset:

        # Prétraitement
        with dt[0]:
            # même prétraitement pour les deux modèles
            im, ims = preprocess_image(im, model_corner)

        # Inférence avec les deux modèles
        with dt[1]:
            pred1_corner = model_prediction(
                model_corner, im, ims, augment, visualize)
            pred2_pieces = model_prediction(
                model_pieces, im, ims, augment, visualize)

        # Application du NMS
        with dt[2]:
            pred1_corner = non_max_suppression(
                pred1_corner, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            pred2_pieces = non_max_suppression(
                pred2_pieces, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Traitement des prédictions de chaque modèle
        for i, (corner_boxes, pieces_boxes) in enumerate(zip(pred1_corner, pred2_pieces)):  # pour chaque image
            seen += 1
            if webcam:  # si c'est une webcam
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            # Chemins de sauvegarde
            p = Path(p)
            save_path = str(save_dir / p.name)

            # Traitement des prédictions du modèle 1
            if len(corner_boxes):
                corner_boxes[:, :4] = scale_boxes(
                    im.shape[2:], corner_boxes[:, :4], im0.shape).round()

            # Traitement des prédictions du modèle 2
            if len(pieces_boxes):
                pieces_boxes[:, :4] = scale_boxes(
                    im.shape[2:], pieces_boxes[:, :4], im0.shape).round()

            corner_boxes = corner_boxes.numpy()
            pieces_boxes = pieces_boxes.numpy()

            pieces_boxes = filter_close_pieces(pieces_boxes)
            print('Nombres de pièces détectées : ', len(pieces_boxes))
            # Etape 2 : Maintenant qu'on a les boxes, on utilise la grosse fonction
            # try:
            #     image_annotated = compute_model_outputs_into_best_move(im0.copy(), corner_boxes, pieces_boxes, ChessBoard, stockfish_path, pieces_logo_directory_path,
            #                                                            display_corners=display_corners, display_boxes=display_boxes, display_coordinates=display_coordinates, display_best_moves=display_best_moves,
            #                                                            display_evaluations=display_evaluations, white_color=white_color, black_color=black_color)
            # except Exception:
            #     image_annotated = im0

            ####### GROSSE FONCTION  ##############

            img = im0.copy()
            corners = get_corner_coordinates_from_corner_boxes(corner_boxes)
            # Etape 3 : Calculer les coordinates des pièces à partir de leur box
            # (le milieu selon x, et le MEME milieu selon y)(il ne faut pas prendre le vrai milieu de la box car les pièces sont hautes mais pas larges)
            pieces_coordinates = []
            for box in pieces_boxes:
                delta = abs(box[2]-box[0])/2
                x = int(box[0] + delta)
                y = int(abs(box[3] - delta))
                # De la forme [x, y, proba, classe]
                pieces_coordinates.append([x, y, box[4], box[5]])

            # Etape 3 : Homographie
            # Taille de l'image redressée (après transformation)
            output_size = 512

            try:
                # Redresser l'image et obtenir les nouvelles coordonnées des pièces
                image_redressee, new_pieces_coordinates, M = redresser_image(
                    img.copy(), corners, pieces_coordinates, output_size)
            except Exception:
                image_redressee, new_pieces_coordinates, M = None, None, None
            try:
                # Partie 4 : Choisir et tourner l'image dans le bon sens parmis les 4 possibilités (deux longues sous étapes)
                image_tournee, new_pieces_coordinates, center, angle = mettre_image_dans_le_bon_sens(
                    image_redressee, new_pieces_coordinates, output_size)
            except Exception:
                image_tournee, new_pieces_coordinates, center, angle = None, None, None, None

            # Etape 5 : Déterminer les cases (h4...) des pièces sur l'échequier dans le bon sens
            try:
                pieces_positions = trouver_cases_pieces(
                    new_pieces_coordinates, output_size)
            except Exception:
                pieces_positions = None
            # Etape 6 : Conversion des cases en positions FEN pour être plus par Stockfish
            try:
                fen = coordinates_to_fen(pieces_positions)
            except Exception:
                fen = None
            # Etape 6 : Visualisation d'un échequier virtuel en continu
            # Affichage de l'échiquier virtuel (board_img)  (si c'est une image seulement sinon ça fait trop bugger la webcam)
            if is_image_file:
                if fen is not None:
                    board_img = ChessBoard.plot_chess_board(fen)
                else:
                    # Afficher une position d'échiquier vide
                    board_img = ChessBoard.plot_chess_board(
                        "8/8/8/8/8/8/8/8 w - - 0 1")

                cv2.namedWindow("Virtual Chess Board",
                                cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.imshow("Virtual Chess Board", board_img)

            # Etape 7 : Prédiction du meilleur coup par Stockfish
            try:
                best_move_white, best_move_black = ChessBoard.get_best_moves(
                    fen)
                white_bar_height, black_bar_height = ChessBoard.get_evaluation_bars(
                    fen)
            except Exception:
                best_move_white, best_move_black = "erreur", "erreur"
                white_bar_height, black_bar_height = None, None

            print("Meilleur coup pour les blancs : ", best_move_white)
            print("Meilleur coup pour les noirs : ", best_move_black)
            # Etape 8 : Convertir cette position h4 en coordonnées sur l'image de base (annuler tous les changements de perspectives et rotation de l'image effectuées)
            try:
                coordonnees_best_moves = convert_best_moves_to_original_image_coordinates(
                    best_move_white, best_move_black, center, angle, M, output_size)
            except Exception:
                coordonnees_best_moves = None

            # Etape 9 : Afficher les coordonnées de ces meilleurs coups sur l'image originale
            # image_annotated = place_best_move_on_image(cv2.imread(img_path), coordonnees_best_moves["white"], coordonnees_best_moves["black"])
            image_annotated = display_everything_on_original_image(img.copy(), corners, pieces_boxes, pieces_coordinates, coordonnees_best_moves, white_bar_height, black_bar_height,
                                                                   display_corners=display_corners, display_boxes=display_boxes, display_coordinates=display_coordinates, display_best_moves=display_best_moves, display_evaluations=display_evaluations, white_color=white_color, black_color=black_color)

            # Save results (image with detections)
            if save_img:
                print(f"Image saved at {save_path}")
                if dataset.mode == "image":
                    cv2.imwrite(save_path, image_annotated)
                else:  # 'video' or 'stream'
                    save_video(i, save_path, vid_path,
                               vid_writer, vid_cap, image_annotated)

            # Stream results
            # Affichage de l'image annotée (image_annotated)
            if view_img or save_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL |
                                    cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(
                        str(p), image_annotated.shape[1], image_annotated.shape[0])
                cv2.imshow(str(p), image_annotated)

            # Conditions pour attendre la fermeture manuelle si la source est une image
            if is_image_file:
                # Rester sur l'image jusqu'à appui sur 'q', 'Esc', ou fermeture de la fenêtre
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    # Vérifier si la fenêtre est fermée ou si 'q' ou 'Esc' est pressé
                    if key == ord('q') or key == 27 or cv2.getWindowProperty("Virtual Chess Board", cv2.WND_PROP_VISIBLE) < 1:
                        cv2.destroyWindow("Virtual Chess Board")
                        cv2.destroyWindow(str(p))
                        break
            else:
                # Mise à jour continue pour la webcam
                cv2.waitKey(1)

        # Print time (inference-only)
        LOGGER.info(
            f"{s}{'' if len(pieces_boxes) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights1", type=str, default="models_weights/best_yolo_corner_detection.pt",
                        help="model path for corner detection")
    parser.add_argument("--weights2", type=str, default="models_weights/best_yolov5x_pieces_detection.pt",
                        help="model path for piece detection")
    parser.add_argument("--source", type=str, default="data/images",
                        help="file/dir/URL/glob/screen/0 (webcam)")
    parser.add_argument("--stockfish_path", type=str,
                        default="stockfish/stockfish-windows-x86-64.exe", help="path to Stockfish executable")
    parser.add_argument("--pieces_logo_directory_path", type=str,
                        default="chess_pieces", help="directory path for chess piece logos")
    parser.add_argument("--display_corners", action="store_true",
                        help="display corners on output image")
    parser.add_argument("--display_boxes", action="store_true",
                        help="display boxes on output image")
    parser.add_argument("--display_coordinates", action="store_true",
                        help="display coordinates on output image")
    parser.add_argument("--display_best_moves", action="store_true",
                        help="display best moves on output image")
    parser.add_argument("--display_evaluations", action="store_true",
                        help="display evaluation bars on output image")
    parser.add_argument("--white_color", nargs=3, type=int,
                        default=(255, 0, 0), help="color for white moves in RGB")
    parser.add_argument("--black_color", nargs=3, type=int,
                        default=(0, 0, 255), help="color for black moves in RGB")
    parser.add_argument("--data", type=str, default="data/coco128.yaml",
                        help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs=2,
                        type=int, default=[640, 640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float,
                        default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float,
                        default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000,
                        help="maximum detections per image")
    parser.add_argument("--device", type=str, default="",
                        help="CUDA device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view_img", action="store_true",
                        help="show results in a window")
    parser.add_argument("--nosave", action="store_true",
                        help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int,
                        help="filter by class: --class 0, or --class 0 2 3")
    parser.add_argument("--agnostic_nms", action="store_true",
                        help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true",
                        help="augmented inference")
    parser.add_argument("--project", type=str,
                        default="runs/detect", help="directory to save results")
    parser.add_argument("--name", type=str, default="exp",
                        help="name of sub-directory for results")
    parser.add_argument("--exist_ok", action="store_true",
                        help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true",
                        help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true",
                        help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid_stride", type=int, default=1,
                        help="video frame-rate stride")

    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

# if __name__ == "__main__":
#     # from yolov5.detect import run
#     corner_detection_model_path = 'models_weights/best_yolo_corner_detection.pt'
#     piece_detection_model_path = 'models_weights/best_yolov5x_pieces_detection.pt'
#     img_path = "c:/Users/tomma/OneDrive/Bureau/Mes_projets/Chess_detection/board_exemple_5.jpg"
#     stockfish_path = 'c:/Users/tomma/OneDrive/Bureau/Mes_projets/Chess_detection/stockfish/stockfish-windows-x86-64.exe'
#     pieces_logo_directory_path = 'c:/Users/tomma/OneDrive/Bureau/Mes_projets/Chess_detection/chess_pieces'
#     run(corner_detection_model_path, piece_detection_model_path, source=0, stockfish_path=stockfish_path,
#         pieces_logo_directory_path=pieces_logo_directory_path, display_corners=True, display_boxes=False,
#         display_coordinates=False, display_best_moves=True, display_evaluations=False, white_color=(255, 0, 0),
#         black_color=(0, 0, 255), view_img=True)
