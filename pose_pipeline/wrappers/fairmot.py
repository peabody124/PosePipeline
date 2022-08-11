import os
import cv2
import torch
import logging
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from pose_pipeline.env import add_path

params = {
    "K": 500,
    "arch": "dla_34",
    "batch_size": 12,
    "cat_spec_wh": False,
    "chunk_sizes": [6, 6],
    "conf_thres": 0.2,  # confidence thresh for tracking
    "det_thres": 0.2,  # confidence thresh for detection
    "nms_thres": 0.4,  # iou thresh for nms
    "dense_wh": False,
    "down_ratio": 4,
    "fix_res": True,
    "gpus": [0],
    "gpus_str": "0",
    "head_conv": 256,
    "heads": {"hm": 1, "wh": 4, "id": 128, "reg": 2},
    "hm_weight": 1,
    "id_loss": "ce",
    "id_weight": 1,
    "img_size": (1088, 608),
    "keep_res": False,
    "lr": 0.0001,
    "lr_step": [20],
    "ltrb": True,
    "master_batch_size": 6,
    "mean": [0.408, 0.447, 0.47],
    "metric": "loss",
    "min_box_area": 100,
    "mse_loss": False,
    "nID": 14455,
    "norm_wh": False,
    "not_cuda_benchmark": False,
    "not_prefetch_test": False,
    "not_reg_offset": False,
    "num_classes": 1,
    "num_epochs": 30,
    "num_iters": -1,
    "num_stacks": 1,
    "num_workers": 8,
    "off_weight": 1,
    "pad": 31,
    "print_iter": 0,
    "reg_loss": "l1",
    "reg_offset": True,
    "reid_dim": 128,
    "seed": 317,
    "std": [0.289, 0.274, 0.278],
    "task": "mot",
    "track_buffer": 30,
    "hide_data_time": True,
    "wh_weight": 0.1,
}


def fairmot_bounding_boxes(file_path):
    from pose_pipeline import MODEL_DATA_DIR

    model_config = os.path.join(MODEL_DATA_DIR, "fairmot/fairmot_dla34.pth")

    opt = dataclass()
    for k, v in params.items():
        opt.__setattr__(k, v)
    opt.load_model = model_config

    tracks = []

    cap = cv2.VideoCapture(file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if height > width:
        opt.img_size = (608, 1088)

    with add_path([os.environ["FAIRMOT_PATH"], os.environ["DCNv2_PATH"]]):
        import datasets.dataset.jde as datasets
        from tracking_utils.log import logger
        from tracker.multitracker import JDETracker
        from tracker.basetrack import BaseTrack

        # suppress logging output
        logger.setLevel(logging.INFO)

        # set up the data loader
        dataloader = datasets.LoadVideo(file_path, opt.img_size)
        fps = dataloader.frame_rate

        width = int(dataloader.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(dataloader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # account for the image rescaling the dataloader performs
        xscale = width / dataloader.w
        yscale = height / dataloader.h

        # prevents prior runs increasing track id
        BaseTrack._count = 0

        # and get the tracker
        tracker = JDETracker(opt, fps)

        for frame_id, (path, img, img0) in tqdm(enumerate(dataloader)):

            blob = torch.from_numpy(img).cuda().unsqueeze(0)

            online_targets = tracker.update(blob, img0)

            frame_result = []

            for t in online_targets:
                tlwh = t.tlwh
                vertical = tlwh[2] / tlwh[3] > 1.6

                if True:  # tlwh[2] * tlwh[3] > opt.min_box_area: # and not vertical:
                    x1, y1, w, h = tlwh

                    # Note: this is using the name "TLHW" consistent with the other algorithms in the
                    # pose_pipeline, but is actually ordered X, Y, W, H.
                    frame_result.append(
                        {
                            "track_id": t.track_id,
                            "tlbr": np.array([x1 * xscale, y1 * yscale, (x1 + w) * xscale, (y1 + h) * yscale]),
                            "tlhw": np.array([x1 * xscale, y1 * yscale, w * xscale, h * yscale]),
                            "confidence": t.score,
                        }
                    )

            tracks.append(frame_result)

        del tracker

        return tracks
