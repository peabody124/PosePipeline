import os
import numpy as np
from tqdm import tqdm

from pose_pipeline import MODEL_DATA_DIR, TopDownPerson, VideoInfo
from pose_pipeline.env import add_path


def process_gastnet(key):

    key = key.copy()

    keypoints = (TopDownPerson & key).fetch1("keypoints")
    height, width = (VideoInfo & key).fetch1("height", "width")

    gastnet_files = os.path.join(MODEL_DATA_DIR, "gastnet/")

    with add_path(os.environ["GAST_PATH"]):

        import torch
        from model.gast_net import SpatioTemporalModel, SpatioTemporalModelOptimized1f
        from common.graph_utils import adj_mx_from_skeleton
        from common.skeleton import Skeleton
        from tools.inference import gen_pose
        from tools.preprocess import h36m_coco_format, revise_kpts

        def gast_load_model(rf=27):
            if rf == 27:
                chk = gastnet_files + "27_frame_model.bin"
                filters_width = [3, 3, 3]
                channels = 128
            elif rf == 81:
                chk = gastnet_files + "81_frame_model.bin"
                filters_width = [3, 3, 3, 3]
                channels = 64
            else:
                raise ValueError("Only support 27 and 81 receptive field models for inference!")

            skeleton = Skeleton(
                parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31],
            )
            adj = adj_mx_from_skeleton(skeleton)

            model_pos = SpatioTemporalModel(
                adj, 17, 2, 17, filter_widths=filters_width, channels=channels, dropout=0.05
            )

            checkpoint = torch.load(chk)
            model_pos.load_state_dict(checkpoint["model_pos"])

            if torch.cuda.is_available():
                model_pos = model_pos.cuda()
            model_pos.eval()

            return model_pos

        keypoints_reformat, keypoints_score = keypoints[None, ..., :2], keypoints[None, ..., 2]
        keypoints, scores, valid_frames = h36m_coco_format(keypoints_reformat, keypoints_score)

        re_kpts = revise_kpts(keypoints, scores, valid_frames)
        assert len(re_kpts) == 1

        rf = 27
        model_pos = gast_load_model(rf)

        pad = (rf - 1) // 2  # Padding on each side
        causal_shift = 0

        # Generating 3D poses
        prediction = gen_pose(re_kpts, valid_frames, width, height, model_pos, pad, causal_shift)

    keypoints_3d = np.zeros((keypoints.shape[1], 17, 3))
    keypoints_3d[np.array(valid_frames[0])] = prediction[0]
    keypoints_valid = [i in valid_frames[0] for i in np.arange(keypoints.shape[1])]

    return {"keypoints_3d": keypoints_3d, "keypoints_valid": keypoints_valid}
