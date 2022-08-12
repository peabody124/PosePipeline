import os
import cv2
import numpy as np
from tqdm import tqdm
import mmtrack.apis


def mmtrack_bounding_boxes(file_path, method="tracktor"):

    from pose_pipeline import MODEL_DATA_DIR

    if method == "tracktor":
        model_config = os.path.join(
            MODEL_DATA_DIR, "mmtracking/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-private.py"
        )
    elif method == "deepsort":
        model_config = os.path.join(
            MODEL_DATA_DIR, "mmtracking/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py"
        )
    elif method == "bytetrack":
        model_config = os.path.join(
            MODEL_DATA_DIR, "mmtracking/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private.py"
        )
    elif method == "qdtrack":
        model_config = os.path.join(
            MODEL_DATA_DIR, "mmtracking/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_crowdhuman_mot17-private-half.py"
        )
    else:
        raise Exception(f"Unknown config file for MMTrack method {method}")
    model = mmtrack.apis.init_model(model_config)

    cap = cv2.VideoCapture(file_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracks = []

    for frame_id in tqdm(range(video_length)):
        ret, frame = cap.read()

        if ret != True or frame is None:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = mmtrack.apis.inference_mot(model, frame, frame_id)

        assert len(result["track_bboxes"]) == 1
        track_results = result["track_bboxes"][0]

        tracks.append(
            [
                {
                    "track_id": int(x[0]),
                    "tlbr": x[1:5],
                    "tlhw": np.array([x[1], x[2], x[3] - x[1], x[4] - x[2]]),
                    "confidence": x[5],
                }
                for x in track_results
            ]
        )

    return tracks
