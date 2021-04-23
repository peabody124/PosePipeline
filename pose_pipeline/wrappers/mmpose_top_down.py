import os
import cv2
import numpy as np
from tqdm import tqdm
import datajoint as dj
from pose_pipeline import Video, TrackingBbox, PersonBboxValid


def mmpose_top_down_person(key):
    
    from mmpose.apis import init_pose_model, inference_top_down_pose_model
    from tqdm import tqdm

    from pose_pipeline import MODEL_DATA_DIR
    pose_cfg = os.path.join(MODEL_DATA_DIR, 'mmpose/config/top_down/darkpose/coco/hrnet_w48_coco_384x288_dark.py')
    pose_ckpt = os.path.join(MODEL_DATA_DIR, 'mmpose/checkpoints/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth')

    video, tracks, keep_tracks = (Video * TrackingBbox * PersonBboxValid & key).fetch1('video', 'tracks', 'keep_tracks')

    model = init_pose_model(pose_cfg, pose_ckpt)

    cap = cv2.VideoCapture(video)

    results = []
    for idx in tqdm(range(len(tracks))):
        bbox = [t['tlhw'] for t in tracks[idx] if t['track_id'] in keep_tracks]

        # handle the case where person is not tracked in frame
        if len(bbox) == 0:
            results.append(np.zeros((17, 3)))
            continue

        # should match the length of identified person tracks
        ret, frame = cap.read()
        assert ret and frame is not None

        bbox_wrap = {'bbox': bbox[0]}
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
        res = inference_top_down_pose_model(model, frame, [bbox_wrap])[0]
        results.append(res[0]['keypoints'])

    os.remove(video)

    return np.asarray(results)
