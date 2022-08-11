#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .yolo import YOLO
from .deep_sort import preprocessing
from .deep_sort import nn_matching
from .deep_sort.detection import Detection
from .deep_sort.detection_yolo import Detection_YOLO
from .deep_sort.tracker import Tracker
from .tools import generate_detections as gdet


def tracking_bounding_boxes(file_path, outfile=None):

    video_capture = cv2.VideoCapture(file_path)
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if outfile is not None:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(outfile, fourcc, fps, (w, h))

    yolo = YOLO()

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    from pose_pipeline import MODEL_DATA_DIR

    model_filename = os.path.join(MODEL_DATA_DIR, "deep_sort_yolov4/mars-small128.pb")
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    tracks = []
    for frame_id in tqdm(range(video_length)):

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True or frame is None:
            break

        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxes, confidence, classes = yolo.detect_image(image)

        features = encoder(frame, boxes)

        detections = [
            Detection(bbox, confidence, cls, feature)
            for bbox, confidence, cls, feature in zip(boxes, confidence, classes, features)
        ]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        tracks.append(
            [
                {
                    "track_id": t.track_id,
                    "tlhw": t.to_tlwh(),
                    "tlbr": t.to_tlbr(),
                    "time_since_update": t.time_since_update,
                }
                for t in tracker.tracks
            ]
        )

        if outfile is not None:

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(
                    frame,
                    "ID: " + str(track.track_id),
                    (int(bbox[0]), int((bbox[3] + bbox[1]) / 2)),
                    0,
                    1.5e-3 * frame.shape[0],
                    (0, 0, 0),
                    thickness=3,
                )
                cv2.putText(
                    frame,
                    "ID: " + str(track.track_id),
                    (int(bbox[0]), int((bbox[3] + bbox[1]) / 2)),
                    0,
                    1.5e-3 * frame.shape[0],
                    (0, 255, 0),
                    thickness=4,
                )

            for det in detections:
                bbox = det.to_tlbr()
                score = "%.2f" % round(det.confidence * 100, 2) + "%"
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                if len(classes) > 0:
                    cls = det.cls
                    cv2.putText(
                        frame,
                        str(cls) + " " + score,
                        (int(bbox[0]), int(bbox[3])),
                        0,
                        1.5e-3 * frame.shape[0],
                        (0, 255, 0),
                        1,
                    )

            out.write(frame)

    video_capture.release()
    if outfile is not None:
        out.release()

    return tracks
