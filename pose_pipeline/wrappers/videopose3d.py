import os
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

from pose_pipeline import MODEL_DATA_DIR, TopDownPerson, VideoInfo
from pose_pipeline.env import add_path


@dataclass
class VideoPoseArgs:
    causal: bool = False
    architecture: str = "3,3,3,3,3"
    dropout: float = 0.25
    channels: int = 1024
    dense: bool = False


def process_videopose3d(key, batch_size=32, transform_coco=False):

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

    max_dim = max(height, width)
    keypoints = normalize_screen_coordinates(keypoints[:, :, :2], width, height)
    keypoints = keypoints[:, :, :2]
    valid_frames = np.arange(keypoints.shape[0])

    with add_path(os.environ["VIDEOPOSE3D_PATH"]):

        import torch
        from common.generators import ChunkedGenerator
        from common.model import TemporalModelOptimized1f, TemporalModel

        args = VideoPoseArgs()
        filter_widths = [int(x) for x in args.architecture.split(",")]
        model_traj = TemporalModelOptimized1f(
            17, 2, 17, filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels
        )

        checkpoint = os.path.join(
            MODEL_DATA_DIR, "videopose3d/pretrained_h36m_detectron_coco.bin"
        )  # pretrained_h36m_cpn.bin')
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

        model_traj.load_state_dict(checkpoint["model_pos"])

        with torch.no_grad():
            model_traj.eval()

        receptive_field = model_traj.receptive_field()
        pad = (receptive_field - 1) // 2  # Padding on each side
        causal_shift = pad if args.causal else 0

        gen = ChunkedGenerator(
            batch_size=batch_size,
            cameras=None,
            poses_3d=None,
            poses_2d=[keypoints[:, :, :2]],
            chunk_length=1,
            shuffle=False,
            pad=pad,
            causal_shift=causal_shift,
        )

        results = []
        with torch.no_grad():
            for sample in tqdm(gen.next_epoch()):
                sample = sample[2]
                sample = torch.from_numpy(sample.astype("float32")).contiguous()
                out = model_traj(sample)

                results.append(out.detach().cpu().numpy()[:, 0, ...])
        results = np.concatenate(results, axis=0)  # / 1000.0

        keypoints_3d = np.zeros((N, 17, 3))
        keypoints_3d[valid_frames] = results
        keypoints_valid = [i in valid_frames.tolist() for i in np.arange(keypoints.shape[0])]

        return {"keypoints_3d": keypoints_3d, "keypoints_valid": keypoints_valid}
