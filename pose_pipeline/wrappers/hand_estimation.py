import os
import cv2
import numpy as np
import datajoint as dj
from pose_pipeline import Video
from pose_pipeline.pipeline import HandBbox
from tqdm import tqdm


def mmpose_HPE(key, method="RTMPoseHand5"):

    from pose_pipeline import MODEL_DATA_DIR

    from mmpose.apis import inference_topdown, init_model
    from mmpose.evaluation.functional import nms

    if method == "RTMPoseHand5":
        pose_model_cfg = os.path.join(MODEL_DATA_DIR,"mmpose/config/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py")
        pose_model_ckpt = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"
    elif method == "RTMPoseCOCO":
        pose_model_cfg = os.path.join(
            MODEL_DATA_DIR,"mmpose/config/hand_2d_keypoint/rtmpose/coco_wholebody_hand/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py",
        )
        pose_model_ckpt = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody-hand_pt-aic-coco_210e-256x256-99477206_20230228.pth"
    elif method == "freihand":
        pose_model_cfg = os.path.join(
            MODEL_DATA_DIR,
            "mmpose/config/hand_2d_keypoint/topdown_heatmap/freihand2d/td-hm_res50_8xb64-100e_freihand2d-224x224.py",
        )
        pose_model_ckpt = "https://download.openmmlab.com/mmpose/hand/resnet/res50_freihand_224x224-ff0799bc_20200914.pth"
    elif method == "HRNet_dark":
        pose_model_cfg = os.path.join(
            MODEL_DATA_DIR,
            "mmpose/config/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py",
        )
        pose_model_ckpt = "https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth"
    elif method == "HRNet_udp":
        pose_model_cfg = os.path.join(
            MODEL_DATA_DIR,
            "mmpose/config/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_udp-8xb64-210e_onehand10k-256x256.py",
        )
        pose_model_ckpt = "https://download.openmmlab.com/mmpose/hand/udp/hrnetv2_w18_onehand10k_256x256_udp-0d1b515d_20210330.pth"

    device = "cuda"
    model = init_model(pose_model_cfg, pose_model_ckpt, device=device)

    video = Video.get_robust_reader(
        key, return_cap=False
    )  # returning video allows deleting it
    bboxes = (HandBbox & key).fetch1("bboxes")

    cap = cv2.VideoCapture(video)
    results = []
    for bbox in tqdm(bboxes):
        ret, frame = cap.read()
        assert ret and frame is not None
        # run the frame through the model
        # Pose_results includes the number of detections
        pose_results = inference_topdown(model, frame, bbox)
        # get prediction instances from mmpose results
        num_hands = len(pose_results)
        keypoints_2d = []
        for i in range(num_hands):
            # pred_instances includes the scores as well as keypoint
            pred_instances = pose_results[i].pred_instances
            keypoints = pred_instances.keypoints
            keypoint_scores = pred_instances.keypoint_scores
            # concat scores and keypoints(flatten)
            keypoints_2d.append(
                np.concatenate((keypoints[0, :, :], keypoint_scores.T), axis=-1)
            )
        results.append(np.concatenate(keypoints_2d, axis=0))

    cap.release()
    os.remove(video)

    return np.array(results)


def overlay_hand_keypoints(video, output_file, keypoints, bboxes):
    """Process a video and create overlay of keypoints

    Args:
    video (str): filename for source (from key)
    output_file (str): output filename
    keypoints (list): list of list of keypoints
    """
    from pose_pipeline.utils.visualization import draw_keypoints

    # Get video details
    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_size = (int(w), int(h))

    # set writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, output_size)
    # process every frame
    for frame_idx in tqdm(range(total_frames)):
        success, frame = cap.read()
        if not success:
            break
        keypoints_2d = keypoints[frame_idx][:, :]
        frame = draw_keypoints(frame, keypoints_2d, threshold=0.2)
        for bbox in bboxes[frame_idx]:
            frame = cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2,
            )  # Green color, 2 pixel thickness

        out.write(frame)
    # remove
    out.release()
    cap.release()