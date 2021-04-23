import os
import cv2
import numpy as np
from tqdm import tqdm
import mmtrack.apis

def mmtrack_bounding_boxes(file_path):
    
    from pose_pipeline import MODEL_DATA_DIR
    model_config = os.path.join(MODEL_DATA_DIR, 'mmtracking/mot/tracktor/tracktor_faster-rcnn_r50_fpn_4e_mot17-private.py')
    
    model = mmtrack.apis.init_model(model_config)

    cap = cv2.VideoCapture(file_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    tracks = []
    
    for frame_id in tqdm(range(video_length)):
        ret, frame = cap.read()
        
        if ret != True or frame is None:
             break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        result = mmtrack.apis.inference_mot(model, frame, frame_id)
        
        assert len(result['track_results']) == 1
        track_results = result['track_results'][0]
        
        tracks.append(
            [{'track_id': int(x[0]),
              'tlbr': x[1:5], 
              'tlhw': np.array([x[1], x[2], x[3]-x[1], x[4]-x[2]]), 
              'confidence': x[5]} for x in track_results])
        
    return tracks