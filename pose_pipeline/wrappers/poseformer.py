import os
import numpy as np
from tqdm import tqdm

from pose_pipeline import MODEL_DATA_DIR, TopDownPerson, VideoInfo
from pose_pipeline.env import add_path


def process_liftformer(key):

    keypoints = (TopDownPerson & key).fetch1("keypoints")
    height, width = (VideoInfo & key).fetch1("height", "width")

    poseformer_files = os.path.join(os.path.split(__file__)[0], "../3rdparty/poseformer/")

    receptive_field = 81
    num_joints = 17

    def coco_h36m(keypoints):
        # adopted from https://github.com/fabro66/GAST-Net-3DPoseEstimation/blob/97a364affe5cd4f68fab030e0210187333fff25e/tools/mpii_coco_h36m.py#L20
        # MIT License

        spple_keypoints = [10, 8, 0, 7]
        h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
        coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        temporal = keypoints.shape[0]
        keypoints_h36m = np.zeros_like(keypoints, dtype=np.float32)
        htps_keypoints = np.zeros((temporal, 4, 2), dtype=np.float32)

        # htps_keypoints: head, thorax, pelvis, spine
        htps_keypoints[:, 0, 0] = np.mean(keypoints[:, 1:5, 0], axis=1, dtype=np.float32)
        htps_keypoints[:, 0, 1] = np.sum(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]
        htps_keypoints[:, 1, :] = np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)
        htps_keypoints[:, 1, :] += (keypoints[:, 0, :] - htps_keypoints[:, 1, :]) / 3

        htps_keypoints[:, 2, :] = np.mean(keypoints[:, 11:13, :], axis=1, dtype=np.float32)
        htps_keypoints[:, 3, :] = np.mean(keypoints[:, [5, 6, 11, 12], :], axis=1, dtype=np.float32)

        keypoints_h36m[:, spple_keypoints, :] = htps_keypoints
        keypoints_h36m[:, h36m_coco_order, :] = keypoints[:, coco_order, :]

        keypoints_h36m[:, 9, :] -= (
            keypoints_h36m[:, 9, :] - np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)
        ) / 4
        keypoints_h36m[:, 7, 0] += 2 * (
            keypoints_h36m[:, 7, 0] - np.mean(keypoints_h36m[:, [0, 8], 0], axis=1, dtype=np.float32)
        )
        keypoints_h36m[:, 8, 1] -= (
            (np.mean(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]) * 2 / 3
        )

        return keypoints_h36m

    # reformat keypoints from coco detection to the input of the lifting
    keypoints = coco_h36m(keypoints[..., :2])
    keypoints = keypoints / np.array([height, width])[None, None, :]

    # reshape into temporal frames. shifted in time as we want to estimate for all
    # time and PoseFormer only produces the central timepoint
    dat = []
    for i in range(keypoints.shape[0] - receptive_field + 1):
        dat.append(keypoints[i : i + receptive_field, :, :2])
    dat = np.stack(dat, axis=0)

    with add_path(os.environ["POSEFORMER_PATH"]):

        import torch
        import torch.nn as nn
        from common.model_poseformer import PoseTransformer

        poseformer = PoseTransformer(
            num_frame=receptive_field,
            num_joints=num_joints,
            in_chans=2,
            embed_dim_ratio=32,
            depth=4,
            num_heads=8,
            mlp_ratio=2.0,
            qkv_bias=True,
            qk_scale=None,
            drop_path_rate=0.1,
        )

        poseformer = nn.DataParallel(poseformer)

        poseformer.cuda()
        chk = os.path.join(poseformer_files, "detected81f.bin")

        checkpoint = torch.load(chk, map_location=lambda storage, loc: storage)
        poseformer.load_state_dict(checkpoint["model_pos"], strict=False)

        kp3d = []
        for idx in range(dat.shape[0]):
            frame = torch.Tensor(dat[None, idx]).cuda()
            kp3d.append(poseformer.forward(frame).cpu().detach().numpy()[:, 0, ...])

        del poseformer
        torch.cuda.empty_cache()

        kp3d = np.concatenate([np.zeros((40, 17, 3)), *kp3d, np.zeros((40, 17, 3))], axis=0)

    key["keypoints_3d"] = kp3d
    return key
