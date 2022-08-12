from pose_pipeline import MODEL_DATA_DIR, TopDownPerson
from pose_pipeline.env import add_path
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import os


@dataclass
class RIEArgs:
    causal: bool = False
    architecture: str = "3,3,3,3,3"
    dropout: float = 0.2
    channels: int = 256
    latent_features_dim: int = 512
    dense: bool = False
    stage: int = 1


def process_rie(key, batch_size=32, transform_coco=False):

    keypoints = (TopDownPerson & key).fetch1("keypoints")
    N = keypoints.shape[0]

    if transform_coco:
        with add_path(os.environ["GAST_PATH"]):
            from tools.preprocess import h36m_coco_format, revise_kpts

            keypoints_reformat, keypoints_score = keypoints[None, ..., :2], keypoints[None, ..., 2]
            keypoints_reformat, scores, valid_frames = h36m_coco_format(keypoints_reformat, keypoints_score)
            keypoints_reformat = revise_kpts(keypoints_reformat, scores, valid_frames)[0]

            valid_frames = np.array(valid_frames[0])
            keypoints = keypoints_reformat[valid_frames]

    else:
        valid_frames = np.arange(keypoints.shape[0])

    with add_path(os.environ["RIE_PATH"]):

        import torch
        from common.generators import Evaluate_Generator
        from common.skeleton import Skeleton
        from common.model import RIEModel

        skeleton = Skeleton(
            parents=[
                -1,
                0,
                1,
                2,
                3,
                4,
                0,
                6,
                7,
                8,
                9,
                0,
                11,
                12,
                13,
                14,
                12,
                16,
                17,
                18,
                19,
                20,
                19,
                22,
                12,
                24,
                25,
                26,
                27,
                28,
                27,
                30,
            ],
            joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
            joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31],
        )

        args = RIEArgs()

        filter_widths = [int(x) for x in args.architecture.split(",")]

        model_pos = RIEModel(
            17,
            2,
            skeleton.num_joints(),
            filter_widths=filter_widths,
            causal=args.causal,
            dropout=args.dropout,
            channels=args.channels,
            latten_features=args.latent_features_dim,
            dense=args.dense,
            is_train=False,
            Optimize1f=True,
            stage=args.stage,
        )

        checkpoint = os.path.join(MODEL_DATA_DIR, "rie/cpn_pretrained.bin")
        checkpoint_p = torch.load(checkpoint, map_location=lambda storage, loc: storage)

        pretrain_dict = checkpoint_p["model_pos"]
        temp = pretrain_dict.items()
        model_dict = model_pos.state_dict()
        state_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict.keys()}
        state_dict = {k: v for i, (k, v) in enumerate(state_dict.items()) if i < 317}

        model_dict.update(state_dict)
        model_pos.load_state_dict(model_dict)

        with torch.no_grad():
            model_pos.eval()

        receptive_field = model_pos.receptive_field()
        pad = (receptive_field - 1) // 2  # Padding on each side
        causal_shift = pad if args.causal else 0

        gen = Evaluate_Generator(
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
                out = model_pos(sample)

                results.append(out.detach().cpu().numpy()[:, 0, ...])
        results = np.concatenate(results, axis=0) / 1000.0

        keypoints_3d = np.zeros((N, 17, 3))
        keypoints_3d[valid_frames] = results
        keypoints_valid = [i in valid_frames.tolist() for i in np.arange(keypoints.shape[0])]

        return {"keypoints_3d": keypoints_3d, "keypoints_valid": keypoints_valid}
