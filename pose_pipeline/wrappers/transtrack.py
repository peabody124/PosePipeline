import os
import cv2
import logging
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from pose_pipeline.env import add_path

params = {
    "dataset_file": "mot",
    "device": "cuda",
    "hidden_dim": 256,
    "position_embedding": "sine",
    "lr_backbone": 2e-5,
    "masks": False,
    "num_feature_levels": 4,
    "backbone": "resnet50",
    "dilation": False,
    "nheads": 8,
    "enc_layers": 6,
    "dec_layers": 6,
    "dim_feedforward": 1024,
    "dropout": 0.1,
    "dec_n_points": 4,
    "enc_n_points": 4,
    "two_stage": False,
    "num_queries": 500,
    "checkpoint_enc_ffn": False,
    "checkpoint_dec_ffn": False,
    "aux_loss": True,
    "with_box_refine": True,
    "set_cost_class": 2,
    "set_cost_bbox": 5,
    "set_cost_giou": 2,
    "cls_loss_coef": 2,
    "bbox_loss_coef": 5,
    "giou_loss_coef": 2,
    "focal_alpha": 0.25,
    "track_thresh": 0.4,
}


def transtrack_bounding_boxes(file_path):

    import torch
    from torchvision import transforms

    from pose_pipeline import MODEL_DATA_DIR

    model_file = os.path.join(MODEL_DATA_DIR, "transtrack/671mot17_crowdhuman_mot17.pth")

    opt = dataclass()
    for k, v in params.items():
        opt.__setattr__(k, v)

    cap = cv2.VideoCapture(file_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    img_size = torch.tensor([[height, width]]).to("cuda")

    tracks = []

    with add_path(os.environ["TRANSTRACK_PATH"]):

        from models import Tracker
        from models import build_tracktrain_model

        # load model
        model, criterion, postprocessors = build_tracktrain_model(opt)
        model.eval()
        checkpoint = torch.load(model_file, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.to("cuda")

        # transform to preprocess images
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        # set up tracker
        tracker = Tracker(params["track_thresh"])
        tracker.reset_all()

        tracks = []

        for i in tqdm(range(frames)):

            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to read a frame")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = transform(frame)[None, ...].to("cuda")

            outputs = model(frame)
            results = postprocessors["bbox"](outputs[0], img_size)

            if i == 0:
                res_track = tracker.init_track(results[0])
            else:
                res_track = tracker.step(results[0])

            del frame
            del outputs

            tracks.append(res_track)

        del tracker
        del model

        def parse_result(x):
            return {
                "track_id": x["tracking_id"],
                "tlbr": np.array(x["bbox"]),
                "tlhw": np.array([*x["bbox"][:2], x["bbox"][2] - x["bbox"][0], x["bbox"][3] - x["bbox"][1]]),
                "confidence": x["score"],
            }

        return [[parse_result(x) for x in frame] for frame in tracks]
