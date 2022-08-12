import os
import cv2
import numpy as np
from tqdm import tqdm
import datajoint as dj
from pose_pipeline import VideoInfo, TopDownPerson


def mmaction_skeleton_action_person(key, device="cuda", stride=1):

    import torch
    import mmcv
    from mmcv.runner import load_checkpoint
    from mmaction.datasets.pipelines import Compose
    from mmaction.models import build_detector, build_model, build_recognizer

    # fetch data
    keypoints = (TopDownPerson & key).fetch1("keypoints")
    img_shape = (h, w) = (VideoInfo & key).fetch1("height", "width")

    # prepare action recognition model
    skeleton_config_file = (
        "/home/jcotton/projects/pose/mmaction2/configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint.py"
    )
    skeleton_stdet_checkpoint = "https://download.openmmlab.com/mmaction/skeleton/posec3d/posec3d_ava.pth"
    label_map = "/home/jcotton/projects/pose/mmaction2/tools/data/ava/label_map.txt"

    def load_label_map(file_path):
        """Load Label Map.
        Args:
            file_path (str): The file path of label map.
        Returns:
            dict: The label map (int -> label name).
        """
        lines = open(file_path).readlines()
        lines = [x.strip().split(": ") for x in lines]
        return {int(x[0]): x[1] for x in lines}

    label_map = load_label_map(label_map)
    num_class = np.max([k for k in label_map.keys()])

    skeleton_config = mmcv.Config.fromfile(skeleton_config_file)
    skeleton_config.model.cls_head.num_classes = num_class + 1  # for AVA

    skeleton_pipeline = Compose(skeleton_config.test_pipeline)
    skeleton_stdet_model = build_model(skeleton_config.model)
    load_checkpoint(skeleton_stdet_model, skeleton_stdet_checkpoint, map_location="cpu")
    skeleton_stdet_model.to(device)
    skeleton_stdet_model.eval()

    # analyze video with rolling window. can increase stride to speed things up
    # but need to account for the fact the time steps won't match
    clip_len = skeleton_config.test_pipeline[0]["clip_len"]

    results = []
    for start in tqdm(range(0, keypoints.shape[0] - clip_len - 1, stride)):
        skeleton_imgs = skeleton_pipeline(
            {
                "keypoint": keypoints[None, start : start + clip_len, :, :2],
                "keypoint_score": keypoints[None, start : start + clip_len, :, 2],
                "total_frames": 10,
                "start_index": 0,
                "modality": "Pose",
                "label": -1,
                "img_shape": img_shape,
            }
        )["imgs"][None]

        skeleton_imgs = skeleton_imgs.to(device)

        with torch.no_grad():
            output = skeleton_stdet_model(return_loss=False, imgs=skeleton_imgs)[0]
            results.append(output)

    def top5(x):
        ind = np.argpartition(x, -5)[-5:]
        ind = ind[np.argsort(-x[ind])]
        return [(label_map[y], x[y]) for y in ind]

    key["top5"] = [top5(x) for x in results]
    key["action_scores"] = np.array(results)
    key["label_map"] = label_map
    key["action_window_len"] = clip_len
    key["stride"] = stride

    return key


class TopDownPersonVideo(dj.Computed):
    definition = """
    -> TopDownPerson
    -> BlurredVideo
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """

    def make(self, key):

        from pose_pipeline.utils.visualization import video_overlay, draw_keypoints

        video = (BlurredVideo & key).fetch1("output_video")
        keypoints = (TopDownPerson & key).fetch1("keypoints")

        bbox_fn = PersonBbox.get_overlay_fn(key)

        def overlay_fn(image, idx):
            image = draw_keypoints(image, keypoints[idx])
            image = bbox_fn(image, idx)
            return image

        fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        video_overlay(video, out_file_name, overlay_fn, downsample=1)

        key["output_video"] = out_file_name

        self.insert1(key)

        os.remove(out_file_name)
        os.remove(video)
