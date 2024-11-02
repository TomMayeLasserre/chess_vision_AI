import cv2
import torch
from pathlib import Path


def model_prediction(model, im, ims, augment, visualize):
    if model.xml and im.shape[0] > 1:
        pred = None
        for image in ims:
            if pred is None:
                pred = model(image, augment=augment,
                             visualize=visualize).unsqueeze(0)
            else:
                pred = torch.cat(
                    (pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
        pred = [pred, None]
    else:
        pred = model(im, augment=augment, visualize=visualize)
    return pred


def preprocess_image(im, model):
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
        return im, None
    if model.xml and im.shape[0] > 1:
        ims = torch.chunk(im, im.shape[0], 0)
        return im, ims
    else:
        return im, None


def save_video(i, save_path, vid_path, vid_writer, vid_cap, im0):
    if vid_path[i] != save_path:  # new video
        vid_path[i] = save_path
        if isinstance(vid_writer[i], cv2.VideoWriter):
            vid_writer[i].release()  # release previous video writer
        if vid_cap:  # video
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:  # stream
            fps, w, h = 30, im0.shape[1], im0.shape[0]
        # force *.mp4 suffix on results videos
        save_path = str(Path(save_path).with_suffix(".mp4"))
        vid_writer[i] = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    vid_writer[i].write(im0)
