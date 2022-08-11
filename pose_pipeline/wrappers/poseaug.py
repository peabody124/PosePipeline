import os
import numpy as np
from tqdm import tqdm

from pose_pipeline import MODEL_DATA_DIR, TopDownPerson, VideoInfo, LiftingPerson
from pose_pipeline.env import add_path


def get_keypoints(key, normalize=True, transform_coco=True):

    keypoints = (TopDownPerson & key).fetch1("keypoints")
    height, width = (VideoInfo & key).fetch1("height", "width")

    N = keypoints.shape[0]

    def normalize_screen_coordinates(X, w, h):
        assert X.shape[-1] == 2

        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        if w > h:
            return X / w * 2 - [1, h / w]
        else:
            return X / h * 2 - [w / h, 1]

    if normalize:
        max_dim = max(height, width)
        keypoints_score = keypoints[None, ..., 2]
        keypoints = normalize_screen_coordinates(keypoints[:, :, :2], width, height)
    else:
        keypoints_score = keypoints[None, ..., 2]
        keypoints = keypoints[:, :, :2]

    if transform_coco:
        with add_path(os.environ["GAST_PATH"]):
            from tools.preprocess import h36m_coco_format, revise_kpts

            keypoints = keypoints[None, ...]
            keypoints_reformat, scores, valid_frames = h36m_coco_format(keypoints, keypoints_score)
            keypoints_reformat = revise_kpts(keypoints_reformat, scores, valid_frames)[0]

            valid_frames = np.array(valid_frames[0])
            keypoints = keypoints_reformat[valid_frames]

    idx_keep = np.array([i for i in range(keypoints.shape[1]) if LiftingPerson.joint_names()[i] != "Nose"])
    keypoints = keypoints[:, idx_keep]

    return keypoints


def process_poseaug(key):

    checkpoint = os.path.join(MODEL_DATA_DIR, "poseaug/ckpt_best_dhp_p1.pth.tar")

    with add_path(os.environ["POSEAUG_PATH"]):

        import torch
        from models_baseline.models_st_gcn.st_gcn_single_frame_test import WrapSTGCN

        model_pos = WrapSTGCN(p_dropout=0.0)
        tmp_ckpt = torch.load(checkpoint)
        model_pos.load_state_dict(tmp_ckpt["model_pos"])
        model_pos.cuda()
        model_pos.eval()

        keypoints = get_keypoints(key)
        keypoints = torch.from_numpy(keypoints[:, :, :2]).cuda().float()
        keypoints_3d = model_pos(keypoints).cpu().detach().numpy()

        keypoints_valid = list(range(keypoints_3d.shape[0]))

        return {"keypoints_3d": keypoints_3d, "keypoints_valid": keypoints_valid}
