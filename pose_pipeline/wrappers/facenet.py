import numpy as np
import tempfile
import cv2
import os

from pose_pipeline import Video


def blur_faces(key, downsample=1):

    from tqdm import trange
    from facenet_pytorch import MTCNN
    from pose_pipeline.utils.visualization import video_overlay

    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        min_face_size=35 // downsample,
        thresholds=[0.4, 0.5, 0.5],
        factor=0.709,
        keep_all=True,
        device="cuda:0",
    )

    video = Video.get_robust_reader(key, return_cap=False)
    cap = cv2.VideoCapture(video)

    def overlay_callback(image, idx, margin=10):
        image = image.copy()

        if downsample is not None and downsample > 0:

            image_ds = cv2.resize(image, (image.shape[1] // downsample, image.shape[0] // downsample))
            boxes, _ = mtcnn.detect(image_ds)

            if boxes is None:
                return image

            boxes = boxes * downsample
        else:
            boxes, _ = mtcnn.detect(image)

        if boxes is None:
            return image

        for y1, x1, y2, x2 in boxes.astype(int):
            x1 = np.max([x1 - margin, 0])
            y1 = np.max([y1 - margin, 0])
            x2 = np.min([x2 + margin, image.shape[1]])
            y2 = np.min([y2 + margin, image.shape[0]])

            if (x1 > x2) or (y1 > y2) or (x1 >= image.shape[1]) or (y1 >= image.shape[0]):
                continue

            try:
                image[x1:x2, y1:y2] = cv2.blur(image[x1:x2, y1:y2], (33, 33)) * 0.9
            except:
                print("Bad bounding box")
                print(x1, x2, y1, y2, image.shape)

        return image

    fid, out_file_name = tempfile.mkstemp(suffix=".mp4")
    video_overlay(video, out_file_name, overlay_callback, downsample=1)

    cap.release()
    os.close(fid)

    os.remove(video)

    return out_file_name
