import os
import cv2
import logging
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from pose_pipeline.env import add_path


params = {
    "K": 100,
    "amodel_offset_weight": 1,
    "arch": "dla_34",
    "backbone": "dla34",
    "box_nms": -1,
    "chunk_sizes": [32],
    "clip_len": 2,
    "dataset": "coco",
    "debug": 0,
    "debugger_theme": "white",
    "deform_kernel_size": 3,
    "dense_reg": 1,
    "dep_weight": 1,
    "depth_scale": 1,
    "dim_weight": 1,
    "dla_node": "dcn",
    "down_ratio": 4,
    "efficient_level": 0,
    "embedding": False,
    "exp_id": "default",
    "fix_res": True,
    "fix_short": -1,
    "flip_test": False,
    "fp_disturb": 0,
    "gpus": [0],
    "gpus_str": "0",
    "head_conv": {"hm": [256], "reg": [256], "wh": [256], "ltrb_amodal": [256]},
    "head_kernel": 3,
    "heads": {"hm": 1, "reg": 2, "wh": 2, "ltrb_amodal": 4},
    "hungarian": False,
    "inference": True,
    "input_h": 480,
    "input_res": 864,
    "input_w": 864,
    "lost_disturb": 0,
    "map_argoverse_id": False,
    "max_age": -1,
    "max_frame_dist": 3,
    "model_output_list": False,
    "msra_outchannel": 256,
    "nID": -1,
    "neck": "dlaup",
    "new_thresh": 0.5,
    "nms": False,
    "no_pause": False,
    "non_block_test": False,
    "num_classes": 1,
    "num_head_conv": 1,
    "num_layers": 101,
    "num_stacks": 1,
    "out_thresh": 0.5,
    "output_h": 120,
    "output_res": 216,
    "output_w": 216,
    "overlap_thresh": 0.05,
    "pad": 31,
    "pre_hm": True,
    "pre_img": True,
    "pre_thresh": 0.5,
    "prior_bias": -4.6,
    "public_det": False,
    "reg_loss": "l1",
    "reset_hm": False,
    "resize_video": True,
    "reuse_hm": False,
    "save_video": False,
    "scale": 0,
    "seg": False,
    "seg_feat_channel": 8,
    "skip_first": -1,
    "task": "tracking",
    "test_focal_length": -1,
    "test_scales": [1.0],
    "track_thresh": 0.5,
    "tracking": True,
    "trades": True,
    "window_size": 20,
    "zero_pre_hm": False,
    "zero_tracking": False,
}


def trades_bounding_boxes(file_path):

    import torch
    from torchvision import transforms

    from pose_pipeline import MODEL_DATA_DIR

    model_file = os.path.join(MODEL_DATA_DIR, "trades/crowdhuman.pth")
    print(model_file)
    opt = dataclass()
    for k, v in params.items():
        opt.__setattr__(k, v)

    opt.load_model = model_file

    cap = cv2.VideoCapture(file_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if height > width:
        # change configuration to better handle portrait mode
        opt.input_h = 864
        opt.input_w = 480
        opt.input_res = 864  # unchanged
        opt.output_h = 216
        opt.output_w = 120
        opt.output_res = 216  # unchanged

    tracks = []

    with add_path(os.environ["TRADES_PATH"]):
        from detector import Detector

        detector = Detector(opt)

        tracks = []

        for i in tqdm(range(frames)):

            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to read a frame")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            results = detector.run(frame)

            tracks.append(results)

        del detector

    def parse_result(x):
        return {
            "track_id": x["tracking_id"],
            "tlbr": np.array(x["bbox"]),
            "tlhw": np.array([*x["bbox"][:2], x["bbox"][2] - x["bbox"][0], x["bbox"][3] - x["bbox"][1]]),
            "confidence": x["score"],
        }

    return [[parse_result(x) for x in frame["results"]] for frame in tracks]
