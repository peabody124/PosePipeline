import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from openpose import pyopenpose as op
from pose_estimation.inference import vid_wrapper

OPENPOSE_MODEL_DIR = os.path.join(os.path.split(op.__file__)[0], '../../../models')

faceRectangles = [
    op.Rectangle(330.119385, 277.532715, 48.717274, 48.717274),
    op.Rectangle(24.036991, 267.918793, 65.175171, 65.175171),
    op.Rectangle(151.803436, 32.477852, 108.295761, 108.295761),
]

handRectangles = [
    # Left/Right hands person 0
    [
    op.Rectangle(320.035889, 377.675049, 69.300949, 69.300949),
    op.Rectangle(0., 0., 0., 0.),
    ],
    # Left/Right hands person 1
    [
    op.Rectangle(80.155792, 407.673492, 80.812706, 80.812706),
    op.Rectangle(46.449715, 404.559753, 98.898178, 98.898178),
    ],
    # Left/Right hands person 2
    [
    op.Rectangle(185.692673, 303.112244, 157.587555, 157.587555),
    op.Rectangle(88.984360, 268.866547, 117.818230, 117.818230),
    ]
]

openpose_joints = {
    'OP_NOSE': 0,
    'OP_NECK': 1,
    'OP_RSHOULDER': 2,
    'OP_RELBOW': 3,
    'OP_RWRIST': 4,
    'OP_LSHOULDER': 5,
    'OP_LELBOW': 6,
    'OP_LWRIST': 7,
    'OP_MIDHIP': 8,
    'OP_RHIP': 9,
    'OP_RKNEE': 10,
    'OP_RANKLE': 11,
    'OP_LHIP': 12,
    'OP_LKNEE': 13,
    'OP_LANKLE': 14,
    'OP_REYE': 15,
    'OP_LEYE': 16,
    'OP_LBIGTOE': 19,
    'OP_LSMALLTOE': 20,
    'OP_LHEEL': 21,
    'OP_RBIGTOE': 22,
    'OP_RSMALLTOE': 23,
    'OP_RHEEL': 24}


class OpenposeParser:
    
    def __init__(self, openpose_model_path=OPENPOSE_MODEL_DIR, 
                 max_people=3, render=True, 
                 results_path=None, hand=False, face=False):
        params = {'model_folder': openpose_model_path,
                  'number_people_max': max_people}

        #params["body"] = 1
        
        self.face = face
        self.hand = hand
        
        if self.face:
            params["face"] = True
            params["face_detector"] = 0
            
        if self.hand:
            params["hand"] = True
            params["hand_detector"] = 1
            
        if results_path is not None:
            params['write_json'] = results_path
        else:
            params['write_json'] = '/tmp/openpose'
            
        if not render:
            params['render_pose'] = 0

        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

    def process_frame(self, im):
        datum = op.Datum()
        datum.cvInputData = im
        datum.faceRectangles = faceRectangles
        datum.handRectangles = handRectangles
        self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        results = {'im': datum.cvOutputData,
                   'keypoints': datum.poseKeypoints}
        if self.hand:
            results['hand_keypoints'] = datum.handKeypoints
        if self.face:
            results['face_keypoints'] = datum.faceKeypoints

        results['pose_ids'] = datum.poseIds;
        results['pose_scores'] = datum.poseScores;
            
        return results
    
    def stop(self):
        self.opWrapper.stop()
        del self.opWrapper
        
        
def openpose_parse_video(video_file):
    
    op = OpenposeParser(face=False, hand=True)  #and=True, face=True)
    results = []

    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames)):
        cont, frame = cap.read()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not cont or frame is None:
            break
        
        res = op.process_frame(frame)
        res.pop('im')
        results.append(res)

    op.stop()
    del op
    
    cap.release()

    return results
