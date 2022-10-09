import os
import cv2
from tqdm import tqdm
from pose_pipeline.env import add_path
from pose_pipeline import Video


openpose_joints = {
    "OP_NOSE": 0,
    "OP_NECK": 1,
    "OP_RSHOULDER": 2,
    "OP_RELBOW": 3,
    "OP_RWRIST": 4,
    "OP_LSHOULDER": 5,
    "OP_LELBOW": 6,
    "OP_LWRIST": 7,
    "OP_MIDHIP": 8,
    "OP_RHIP": 9,
    "OP_RKNEE": 10,
    "OP_RANKLE": 11,
    "OP_LHIP": 12,
    "OP_LKNEE": 13,
    "OP_LANKLE": 14,
    "OP_REYE": 15,
    "OP_LEYE": 16,
    "OP_REAR": 17,
    "OP_LEAR": 18,
    "OP_LBIGTOE": 19,
    "OP_LSMALLTOE": 20,
    "OP_LHEEL": 21,
    "OP_RBIGTOE": 22,
    "OP_RSMALLTOE": 23,
    "OP_RHEEL": 24,
}


class OpenposeParser:
    def __init__(self, openpose_model_path=None, max_people=10, render=True, results_path=None, hand=False, face=False,
                 model_pose='BODY_25', **kwargs):

        from openpose import pyopenpose as op

        if openpose_model_path is None:
            openpose_model_path = os.path.join(os.path.split(op.__file__)[0], "../../../models")

        self.faceRectangles = [
            op.Rectangle(330.119385, 277.532715, 48.717274, 48.717274),
            op.Rectangle(24.036991, 267.918793, 65.175171, 65.175171),
            op.Rectangle(151.803436, 32.477852, 108.295761, 108.295761),
        ]

        self.handRectangles = [
            # Left/Right hands person 0
            [
                op.Rectangle(320.035889, 377.675049, 69.300949, 69.300949),
                op.Rectangle(0.0, 0.0, 0.0, 0.0),
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
            ],
        ]
        params = {"model_folder": openpose_model_path, "number_people_max": max_people}

        params["body"] = 1

        self.face = face
        self.hand = hand
        self.render = render

        if self.face:
            params["face"] = True
            params["face_detector"] = 0

        if self.hand:
            params["hand"] = True
            params["hand_detector"] = 0

        if results_path is not None:
            params["write_json"] = results_path
        else:
            params["write_json"] = "/tmp/openpose"

        if not render:
            params["render_pose"] = 0

        params["model_pose"] = model_pose

        for k, v in kwargs.items():
            print(k, v)
            params[k] = v

        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

    def process_frame(self, im):
        from openpose import pyopenpose as op

        datum = op.Datum()
        datum.cvInputData = im
        datum.faceRectangles = self.faceRectangles
        datum.handRectangles = self.handRectangles
        self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        results = {
            "im": datum.cvOutputData if self.render else None,
            "keypoints": datum.poseKeypoints,
            "hand_keypoints": datum.handKeypoints if self.hand else None,
            "face_keypoints": datum.faceKeypoints if self.face else None,
        }

        results["pose_ids"] = datum.poseIds
        results["pose_scores"] = datum.poseScores

        return results

    def stop(self):
        self.opWrapper.stop()
        del self.opWrapper


def openpose_parse_video(video_file, **kwargs):

    op = OpenposeParser(render=False, **kwargs)
    results = []

    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()

        if not ret or frame is None:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = op.process_frame(frame)
        res.pop("im")
        results.append(res)

    op.stop()
    del op

    cap.release()

    return results


def openpose_process_key(key, **kwargs):

    video = Video.get_robust_reader(key, return_cap=False)

    with add_path(os.path.join(os.environ["OPENPOSE_PATH"], "build/python")):
        from pose_pipeline.wrappers.openpose import openpose_parse_video

    res = openpose_parse_video(video, **kwargs)
    os.remove(video)

    key["keypoints"] = res
    return key