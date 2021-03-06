import os
import cv2
import numpy as np
from tqdm import tqdm
import datajoint as dj
from pose_pipeline import Video, PersonBbox


def mmpose_top_down_person(key):
    
    from mmpose.apis import init_pose_model, inference_top_down_pose_model
    from tqdm import tqdm

    from pose_pipeline import MODEL_DATA_DIR
    pose_cfg = os.path.join(MODEL_DATA_DIR, 'mmpose/config/top_down/darkpose/coco/hrnet_w48_coco_384x288_dark.py')
    pose_ckpt = os.path.join(MODEL_DATA_DIR, 'mmpose/checkpoints/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth')

    bboxes = (PersonBbox & key).fetch1('bbox')
    cap = Video.get_robust_reader(key)

    model = init_pose_model(pose_cfg, pose_ckpt)

    results = []
    for bbox in tqdm(bboxes):

        # should match the length of identified person tracks
        ret, frame = cap.read()
        assert ret and frame is not None

        # handle the case where person is not tracked in frame
        if np.any(np.isnan(bbox)):
            results.append(np.zeros((17, 3)))
            continue

        bbox_wrap = {'bbox': bbox}
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
        res = inference_top_down_pose_model(model, frame, [bbox_wrap])[0]
        results.append(res[0]['keypoints'])

    return np.asarray(results)


def mmpose_whole_body(key):
    # keypoint order can be found in 
    # https://github.com/jin-s13/COCO-WholeBody

    from mmpose.apis import init_pose_model, inference_top_down_pose_model
    from tqdm import tqdm

    from pose_pipeline import MODEL_DATA_DIR
    pose_cfg = os.path.join(MODEL_DATA_DIR, 'mmpose/config/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py')
    pose_ckpt = os.path.join(MODEL_DATA_DIR, 'mmpose/checkpoints/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth')

    bboxes = (PersonBbox & key).fetch1('bbox')
    cap = Video.get_robust_reader(key)

    model = init_pose_model(pose_cfg, pose_ckpt)

    results = []
    for bbox in tqdm(bboxes):

        # should match the length of identified person tracks
        ret, frame = cap.read()
        assert ret and frame is not None

        # handle the case where person is not tracked in frame
        if np.any(np.isnan(bbox)):
            results.append(np.zeros((17, 3)))
            continue

        bbox_wrap = {'bbox': bbox}

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = inference_top_down_pose_model(model, frame, [bbox_wrap])[0]
        results.append(res[0]['keypoints'])

    return np.asarray(results)


def mmpose_bottom_up(key):

    from mmpose.apis import init_pose_model, inference_bottom_up_pose_model
    from tqdm import tqdm

    from pose_pipeline import MODEL_DATA_DIR
    pose_cfg = os.path.join(MODEL_DATA_DIR, 'mmpose/config/bottom_up/higherhrnet/coco/higher_hrnet48_coco_512x512.py')
    pose_ckpt = os.path.join(MODEL_DATA_DIR, 'mmpose/checkpoints/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth')

    model = init_pose_model(pose_cfg, pose_ckpt)

    video = Video.get_robust_reader(key, return_cap=False)
    cap = cv2.VideoCapture(video)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    keypoints = []
    for frame_id in tqdm(range(video_length)):

        # should match the length of identified person tracks
        ret, frame = cap.read()
        assert ret and frame is not None
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
        res = inference_bottom_up_pose_model(model, frame)[0]

        kps = np.stack([x['keypoints'] for x in res], axis=0)
        keypoints.append(kps)

    cap.release()
    os.remove(video)

    return np.asarray(keypoints)
    