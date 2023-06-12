import os
import sys
import cv2
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import shutil

import datajoint as dj

from .utils.keypoint_matching import match_keypoints_to_bbox
from .env import add_path

if "custom" not in dj.config:
    dj.config["custom"] = {}

db_prefix = dj.config["custom"].get("database.prefix", "")

schema = dj.schema(db_prefix + "pose_pipeline")


@schema
class Video(dj.Manual):
    definition = """
    video_project       : varchar(50)
    filename            : varchar(100)
    ---
    video               : attach@localattach    # datajoint managed video file
    start_time          : timestamp(3)          # time of beginning of video, as accurately as known
    import_time  = CURRENT_TIMESTAMP : timestamp
    """

    @staticmethod
    def make_entry(filepath, session_id=None):
        from datetime import datetime
        import os

        _, fn = os.path.split(filepath)
        date = datetime.strptime(fn[:16], "%Y%m%d-%H%M%SZ")
        d = {"filename": fn, "video": filepath, "start_time": date}
        if session_id is not None:
            d.update({"session_id": session_id})
        return d

    @staticmethod
    def get_robust_reader(key, return_cap=True):
        import subprocess
        import tempfile

        # fetch video and place in temp directory
        video = (Video & key).fetch1("video")
        fd, outfile = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        shutil.move(video, outfile)

        video = outfile

        cap = cv2.VideoCapture(video)

        # check all the frames are readable
        expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        completed = True

        def compress(video):
            fd, outfile = tempfile.mkstemp(suffix=".mp4")
            print(f"Unable to read all the fails. Transcoding {video} to {outfile}")
            subprocess.run(["ffmpeg", "-y", "-i", video, "-c:v", "libx264", "-b:v", "1M", outfile])
            os.close(fd)
            return outfile

        for i in range(expected_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()

                video = compress(video)
                cap = cv2.VideoCapture(video)
                break

        if return_cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return cap
        else:
            cap.release()
            return video


@schema
class VideoInfo(dj.Computed):
    definition = """
    -> Video
    ---
    timestamps      : longblob
    delta_time      : longblob
    fps             : float
    height          : int
    width           : int
    num_frames      : int
    """

    def make(self, key, override=False):

        key = key.copy()
        video, start_time = (Video & key).fetch1("video", "start_time")

        cap = cv2.VideoCapture(video)
        key["fps"] = fps = cap.get(cv2.CAP_PROP_FPS)
        key["num_frames"] = frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        key["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        key["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        key["timestamps"] = [start_time + timedelta(0, i / fps) for i in range(frames)]
        key["delta_time"] = [timedelta(0, i / fps).total_seconds() for i in range(frames)]

        cap.release()
        os.remove(video)

        self.insert1(key, allow_direct_insert=override)

    def fetch_timestamps(self):
        assert len(self) == 1, "Restrict to single entity"
        timestamps = self.fetch1("timestamps")
        timestamps = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
        return timestamps


@schema
class BottomUpMethodLookup(dj.Lookup):
    definition = """
    bottom_up_method_name : varchar(50)
    """
    contents = [
        {"bottom_up_method_name": "OpenPose"},
        {"bottom_up_method_name": "OpenPose_BODY25B"},
        {"bottom_up_method_name": "MMPose"},
    ]


@schema
class BottomUpMethod(dj.Manual):
    definition = """
    -> Video
    -> BottomUpMethodLookup
    """


@schema
class BottomUpPeople(dj.Computed):
    definition = """
    -> BottomUpMethod
    ---
    keypoints                   : longblob
    timestamp=CURRENT_TIMESTAMP : timestamp    # automatic timestamp
    """

    def make(self, key):

        if key["bottom_up_method_name"] == "OpenPose":
            from pose_pipeline.wrappers.openpose import openpose_process_key

            params = {"model_pose": "BODY_25", "scale_number": 4, "scale_gap": 0.25}
            key = openpose_process_key(key, **params)
            # to standardize with MMPose, drop other info
            key["keypoints"] = [k["keypoints"] for k in key["keypoints"]]

        elif key["bottom_up_method_name"] == "OpenPose_BODY25B":
            from pose_pipeline.wrappers.openpose import openpose_process_key

            params = {"model_pose": "BODY_25B", "scale_number": 4, "scale_gap": 0.25}
            key = openpose_process_key(key, **params)
            # to standardize with MMPose, drop other info
            key["keypoints"] = [k["keypoints"] for k in key["keypoints"]]

        elif key["bottom_up_method_name"] == "MMPose":
            from .wrappers.mmpose import mmpose_bottom_up

            key["keypoints"] = mmpose_bottom_up(key)

        else:
            raise Exception("Method not implemented")

        self.insert1(key)


@schema
class BottomUpVideo(dj.Computed):
    definition = """
    -> BottomUpPeople
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """

    def make(self, key):

        from pose_pipeline.utils.visualization import video_overlay, draw_keypoints

        video = (BlurredVideo & key).fetch1("output_video")
        keypoints = (BottomUpPeople & key).fetch1("keypoints")

        def get_color(i):
            import numpy as np

            c = np.array([np.cos(i * np.pi / 2), np.cos(i * np.pi / 4), np.cos(i * np.pi / 8)]) * 127 + 127
            return c.astype(int).tolist()

        def overlay_fn(image, idx):
            if keypoints[idx] is None:
                return image
            for person_idx in range(keypoints[idx].shape[0]):
                image = draw_keypoints(image, keypoints[idx][person_idx], color=get_color(person_idx))
            return image

        fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
        video_overlay(video, out_file_name, overlay_fn, downsample=1)
        os.close(fd)

        key["output_video"] = out_file_name

        self.insert1(key)

        os.remove(out_file_name)
        os.remove(video)


@schema
class OpenPose(dj.Computed):
    definition = """
    -> Video
    ---
    keypoints         : longblob
    pose_ids          : longblob
    pose_scores       : longblob
    face_keypoints    : longblob
    hand_keypoints    : longblob
    """

    def make(self, key):

        video = Video.get_robust_reader(key, return_cap=False)

        with add_path(os.path.join(os.environ["OPENPOSE_PATH"], "build/python")):
            from pose_pipeline.wrappers.openpose import openpose_parse_video

            res = openpose_parse_video(video, face=False, hand=True, scale_number=4)

        key["keypoints"] = [r["keypoints"] for r in res]
        key["pose_ids"] = [r["pose_ids"] for r in res]
        key["pose_scores"] = [r["pose_scores"] for r in res]
        key["hand_keypoints"] = [r["hand_keypoints"] for r in res]
        key["face_keypoints"] = [r["face_keypoints"] for r in res]

        self.insert1(key)

        # remove the downloaded video to avoid clutter
        os.remove(video)


@schema
class OpenPoseVideo(dj.Computed):
    definition = """
    -> OpenPose
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """

    def make(self, key):

        from pose_pipeline.utils.visualization import video_overlay, draw_keypoints

        video = (BlurredVideo & key).fetch1("output_video")
        keypoints = (OpenPose & key).fetch1("keypoints")

        def overlay_fn(image, idx):
            if keypoints[idx] is None:
                return image
            for person_idx in range(keypoints[idx].shape[0]):
                image = draw_keypoints(image, keypoints[idx][person_idx])
            return image

        fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        video_overlay(video, out_file_name, overlay_fn, downsample=1)

        key["output_video"] = out_file_name

        self.insert1(key)

        os.remove(out_file_name)
        os.remove(video)


@schema
class BlurredVideo(dj.Computed):
    definition = """
    -> Video
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """

    def make(self, key):

        from .wrappers.facenet import blur_faces

        blurred_video = blur_faces(key)

        key["output_video"] = blurred_video
        self.insert1(key)

        os.remove(blurred_video)


@schema
class TrackingBboxMethodLookup(dj.Lookup):
    definition = """
    tracking_method      : int
    ---
    tracking_method_name : varchar(50)
    """
    contents = [
        {"tracking_method": 0, "tracking_method_name": "DeepSortYOLOv4"},
        {"tracking_method": 1, "tracking_method_name": "MMTrack_tracktor"},
        {"tracking_method": 2, "tracking_method_name": "FairMOT"},
        {"tracking_method": 3, "tracking_method_name": "TransTrack"},
        {"tracking_method": 4, "tracking_method_name": "TraDeS"},
        {"tracking_method": 5, "tracking_method_name": "MMTrack_deepsort"},
        {"tracking_method": 6, "tracking_method_name": "MMTrack_bytetrack"},
        {"tracking_method": 7, "tracking_method_name": "MMTrack_qdtrack"},
    ]


@schema
class TrackingBboxMethod(dj.Manual):
    definition = """
    -> Video
    tracking_method   : int
    ---
    """


@schema
class TrackingBbox(dj.Computed):
    definition = """
    -> TrackingBboxMethod
    ---
    tracks            : longblob
    num_tracks        : int
    """

    def make(self, key):

        video = Video.get_robust_reader(key, return_cap=False)

        if (TrackingBboxMethodLookup & key).fetch1("tracking_method_name") == "DeepSortYOLOv4":
            from pose_pipeline.wrappers.deep_sort_yolov4.parser import tracking_bounding_boxes

            tracks = tracking_bounding_boxes(video)
            key["tracks"] = tracks

        elif (TrackingBboxMethodLookup & key).fetch1("tracking_method_name") in "MMTrack_tracktor":
            from pose_pipeline.wrappers.mmtrack import mmtrack_bounding_boxes

            tracks = mmtrack_bounding_boxes(video, "tracktor")
            key["tracks"] = tracks

        elif (TrackingBboxMethodLookup & key).fetch1("tracking_method_name") == "MMTrack_deepsort":
            from pose_pipeline.wrappers.mmtrack import mmtrack_bounding_boxes

            tracks = mmtrack_bounding_boxes(video, "deepsort")
            key["tracks"] = tracks

        elif (TrackingBboxMethodLookup & key).fetch1("tracking_method_name") == "MMTrack_bytetrack":
            from pose_pipeline.wrappers.mmtrack import mmtrack_bounding_boxes

            tracks = mmtrack_bounding_boxes(video, "bytetrack")
            key["tracks"] = tracks

        elif (TrackingBboxMethodLookup & key).fetch1("tracking_method_name") == "MMTrack_qdtrack":
            from pose_pipeline.wrappers.mmtrack import mmtrack_bounding_boxes

            tracks = mmtrack_bounding_boxes(video, "qdtrack")
            key["tracks"] = tracks

        elif (TrackingBboxMethodLookup & key).fetch1("tracking_method_name") == "FairMOT":
            from pose_pipeline.wrappers.fairmot import fairmot_bounding_boxes

            tracks = fairmot_bounding_boxes(video)
            key["tracks"] = tracks

        elif (TrackingBboxMethodLookup & key).fetch1("tracking_method_name") == "TransTrack":
            from pose_pipeline.wrappers.transtrack import transtrack_bounding_boxes

            tracks = transtrack_bounding_boxes(video)
            key["tracks"] = tracks

        elif (TrackingBboxMethodLookup & key).fetch1("tracking_method_name") == "TraDeS":
            from pose_pipeline.wrappers.trades import trades_bounding_boxes

            tracks = trades_bounding_boxes(video)
            key["tracks"] = tracks

        else:
            os.remove(video)
            raise Exception(f"Unsupported tracking method: {key['tracking_method']}")

        track_ids = np.unique([t["track_id"] for track in tracks for t in track])
        key["num_tracks"] = len(track_ids)

        self.insert1(key)

        # remove the downloaded video to avoid clutter
        if os.path.exists(video):
            os.remove(video)


@schema
class TrackingBboxVideo(dj.Computed):
    definition = """
    -> BlurredVideo
    -> TrackingBbox
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """

    def make(self, key):

        import matplotlib
        from pose_pipeline.utils.visualization import video_overlay

        video = (BlurredVideo & key).fetch1("output_video")
        tracks = (TrackingBbox & key).fetch1("tracks")

        N = len(np.unique([t["track_id"] for track in tracks for t in track]))
        colors = matplotlib.cm.get_cmap("hsv", lut=N)

        def overlay_callback(image, idx):
            image = image.copy()

            for track in tracks[idx]:
                c = colors(track["track_id"])
                c = (int(c[0] * 255.0), int(c[1] * 255.0), int(c[2] * 255.0))

                small = int(5e-3 * np.max(image.shape))
                large = 2 * small

                bbox = track["tlbr"]
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), large)
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), c, small)

                label = str(track["track_id"])
                textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, int(5.0e-3 * image.shape[0]), 4)[0]
                x = int((bbox[0] + bbox[2]) / 2 - textsize[0] / 2)
                y = int((bbox[3] + bbox[1]) / 2 + textsize[1] / 2)
                cv2.putText(image, label, (x, y), 0, 5.0e-3 * image.shape[0], (255, 255, 255), thickness=large)
                cv2.putText(image, label, (x, y), 0, 5.0e-3 * image.shape[0], c, thickness=small)

            return image

        fd, fname = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        video_overlay(video, fname, overlay_callback, downsample=1)

        key["output_video"] = fname

        self.insert1(key)

        # remove the downloaded video to avoid clutter
        os.remove(video)
        os.remove(fname)


@schema
class PersonBboxValid(dj.Manual):
    definition = """
    -> TrackingBbox
    video_subject_id        : int
    ---
    keep_tracks             : longblob
    """


@schema
class PersonBbox(dj.Computed):
    definition = """
    -> PersonBboxValid
    ---
    bbox               : longblob
    present            : longblob
    """

    def make(self, key):

        tracks = (TrackingBbox & key).fetch1("tracks")
        keep_tracks = (PersonBboxValid & key).fetch1("keep_tracks")

        def extract_person_track(tracks):
            def process_timestamp(track_timestep):
                valid = [t for t in track_timestep if t["track_id"] in keep_tracks]
                if len(valid) == 1:
                    return {"present": True, "bbox": valid[0]["tlhw"]}
                else:
                    return {"present": False, "bbox": [0.0, 0.0, 0.0, 0.0]}

            return [process_timestamp(t) for t in tracks]

        LD = main_track = extract_person_track(tracks)
        dict_lists = {k: [dic[k] for dic in LD] for k in LD[0]}

        present = np.array(dict_lists["present"])
        bbox = np.array(dict_lists["bbox"])

        # smooth any brief missing frames
        df = pd.DataFrame(bbox)
        df.iloc[~present] = np.nan
        df = df.fillna(method="bfill", axis=0, limit=2)
        df = df.fillna(method="ffill", axis=0, limit=2)

        # get smoothed version
        key["present"] = ~df.isna().any(axis=1).values
        key["bbox"] = df.values

        self.insert1(key)

    @staticmethod
    def get_overlay_fn(key):

        bboxes = (PersonBbox & key).fetch1("bbox")

        def overlay_fn(image, idx, width=6, color=(255, 255, 255)):
            bbox = bboxes[idx].copy()
            bbox[2:] = bbox[:2] + bbox[2:]
            if np.any(np.isnan(bbox)):
                return image

            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, width)
            return image

        return overlay_fn

    @property
    def key_source(self):
        return PersonBboxValid & "video_subject_id >= 0"


@schema
class DetectedFrames(dj.Computed):
    definition = """
    -> PersonBboxValid
    -> VideoInfo
    ---
    frames_detected        : int
    frames_missed          : int
    fraction_found         : float
    mean_other_people      : float
    median_confidence      : float
    frame_data             : longblob
    """

    def make(self, key):

        if (PersonBboxValid & key).fetch1("video_subject_id") < 0:
            key["frames_detected"] = 0
            key["frames_missed"] = (VideoInfo & key).fetch1("num_frames")

        # compute statistics
        tracks = (TrackingBbox & key).fetch1("tracks")
        keep_tracks = (PersonBboxValid & key).fetch1("keep_tracks")

        def extract_person_stats(tracks):
            def process_timestamp(track_timestep):
                valid = [t for t in track_timestep if t["track_id"] in keep_tracks]
                total_tracks = len(track_timestep)
                if len(valid) == 1:
                    if "confidence" in valid[0].keys():
                        return {"present": True, "confidence": valid[0]["confidence"], "others": total_tracks - 1}
                    else:
                        return {"present": True, "confidence": 1.0, "others": total_tracks - 1}
                else:
                    return {"present": False, "confidence": 0, "others": total_tracks}

            return [process_timestamp(t) for t in tracks]

        stats = extract_person_stats(tracks)
        present = np.array([x["present"] for x in stats])

        key["frames_detected"] = np.sum(present)
        key["frames_missed"] = np.sum(~present)
        key["fraction_found"] = key["frames_detected"] / (key["frames_missed"] + key["frames_detected"])

        if key["frames_detected"] > 0:
            key["median_confidence"] = np.median([x["confidence"] for x in stats if x["present"]])
        else:
            key["median_confidence"] = 0.0
        key["mean_other_people"] = np.nanmean([x["others"] for x in stats])
        key["frame_data"] = stats

        self.insert1(key)

    @property
    def key_source(self):
        return PersonBboxValid & "video_subject_id >= 0"


@schema
class BestDetectedFrames(dj.Computed):
    definition = """
    -> DetectedFrames
    """

    def make(self, key):
        detected_frames = (DetectedFrames & key).fetch("fraction_found", "KEY", as_dict=True)

        best = np.argmax([d["fraction_found"] for d in detected_frames])
        res = detected_frames[best]
        res.pop("fraction_found")
        self.insert1(res)

    @property
    def key_source(self):
        return Video & DetectedFrames


@schema
class BottomUpPerson(dj.Computed):
    definition = """
    -> PersonBbox
    -> BottomUpPeople
    ---
    keypoints        : longblob
    """

    def make(self, key):

        print(key)

        # fetch data
        keypoints = (BottomUpPeople & key).fetch1("keypoints")
        bbox = (PersonBbox & key).fetch1("bbox")

        res = [match_keypoints_to_bbox(bbox[idx], keypoints[idx]) for idx in range(bbox.shape[0])]
        keypoints, _ = list(zip(*res))
        keypoints = np.array(keypoints)
        key["keypoints"] = keypoints

        self.insert1(key)


@schema
class OpenPosePerson(dj.Computed):
    definition = """
    -> PersonBbox
    -> OpenPose
    ---
    keypoints        : longblob
    hand_keypoints   : longblob
    openpose_ids     : longblob
    """

    def make(self, key):

        # fetch data
        keypoints, hand_keypoints = (OpenPose & key).fetch1("keypoints", "hand_keypoints")
        bbox = (PersonBbox & key).fetch1("bbox")

        res = [match_keypoints_to_bbox(bbox[idx], keypoints[idx]) for idx in range(bbox.shape[0])]
        keypoints, openpose_ids = list(zip(*res))

        keypoints = np.array(keypoints)
        openpose_ids = np.array(openpose_ids)

        key["keypoints"] = keypoints
        key["openpose_ids"] = openpose_ids

        key["hand_keypoints"] = []

        for openpose_id, hand_keypoint in zip(openpose_ids, hand_keypoints):
            if openpose_id is None:
                key["hand_keypoints"].append(np.zeros((2, 21, 3)))
            else:
                key["hand_keypoints"].append([hand_keypoint[0][openpose_id], hand_keypoint[1][openpose_id]])
        key["hand_keypoints"] = np.asarray(key["hand_keypoints"])

        self.insert1(key)

    @staticmethod
    def joint_names():
        return [
            "Nose",
            "Sternum",
            "Right Shoulder",
            "Right Elbow",
            "Right Wrist",
            "Left Shoulder",
            "Left Elbow",
            "Left Wrist",
            "Pelvis",
            "Right Hip",
            "Right Knee",
            "Right Ankle",
            "Left Hip",
            "Left Knee",
            "Left Ankle",
            "Right Eye",
            "Left Eye",
            "Right Ear",
            "Left Ear",
            "Left Big Toe",
            "Left Little Toe",
            "Left Heel",
            "Right Big Toe",
            "Right Little Toe",
            "Right Heel",
        ]


@schema
class OpenPosePersonVideo(dj.Computed):
    definition = """
    -> OpenPosePerson
    -> BlurredVideo
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """

    def make(self, key):

        from pose_pipeline.utils.visualization import video_overlay, draw_keypoints

        # fetch data
        keypoints, hand_keypoints = (OpenPosePerson & key).fetch1("keypoints", "hand_keypoints")
        video_filename = (BlurredVideo & key).fetch1("output_video")

        fd, fname = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        video = (BlurredVideo & key).fetch1("output_video")
        keypoints = (OpenPosePerson & key).fetch1("keypoints")

        def overlay(image, idx):
            image = draw_keypoints(image, keypoints[idx])
            image = draw_keypoints(image, hand_keypoints[idx, 0], threshold=0.02)
            image = draw_keypoints(image, hand_keypoints[idx, 1], threshold=0.02)
            return image

        ofd, out_file_name = tempfile.mkstemp(suffix=".mp4")
        os.close(ofd)
        video_overlay(video, out_file_name, overlay, downsample=4)
        key["output_video"] = out_file_name

        self.insert1(key)

        os.remove(out_file_name)
        os.remove(video)


@schema
class TopDownMethodLookup(dj.Lookup):
    definition = """
    top_down_method      : int
    ---
    top_down_method_name : varchar(50)
    """
    contents = [
        {"top_down_method": 0, "top_down_method_name": "MMPose"},
        {"top_down_method": 1, "top_down_method_name": "MMPoseWholebody"},
        {"top_down_method": 2, "top_down_method_name": "MMPoseHalpe"},
        {"top_down_method": 3, "top_down_method_name": "MMPoseHrformerCoco"},
        {"top_down_method": 4, "top_down_method_name": "OpenPose"},
        {"top_down_method": 6, "top_down_method_name": "OpenPose_BODY25B"},
    ]


@schema
class TopDownMethod(dj.Manual):
    definition = """
    -> PersonBbox
    top_down_method    : int
    """


@schema
class TopDownPerson(dj.Computed):
    definition = """
    -> TopDownMethod
    ---
    keypoints          : longblob
    """

    def make(self, key):

        method_name = (TopDownMethodLookup & key).fetch1("top_down_method_name")
        if method_name == "MMPose":
            from .wrappers.mmpose import mmpose_top_down_person

            key["keypoints"] = mmpose_top_down_person(key, "HRNet_W48_COCO")
        elif method_name == "MMPoseWholebody":
            from .wrappers.mmpose import mmpose_top_down_person

            key["keypoints"] = mmpose_top_down_person(key, "HRNet_W48_COCOWholeBody")
        elif method_name == "MMPoseTCFormerWholebody":
            from .wrappers.mmpose import mmpose_top_down_person

            key["keypoints"] = mmpose_top_down_person(key, "HRNet_TCFormer_COCOWholeBody")
        elif method_name == "MMPoseHalpe":
            from .wrappers.mmpose import mmpose_top_down_person

            key["keypoints"] = mmpose_top_down_person(key, "HRNet_W48_HALPE")
        elif method_name == "MMPoseHrformerCoco":
            from .wrappers.mmpose import mmpose_top_down_person

            key["keypoints"] = mmpose_top_down_person(key, "HRFormer_COCO")
        elif method_name == "OpenPose":
            # Manually copying data over to allow this to be used consistently
            # but also take advantage of the logic assigning the OpenPose person as a
            # person of interest
            key["keypoints"] = (OpenPosePerson & key).fetch1("keypoints")
        elif method_name == "OpenPose_BODY25B":
            # Manually copying data over to allow this to be used consistently
            # but also take advantage of the logic assigning the OpenPose person as a
            # person of interest
            key["keypoints"] = (BottomUpPerson & key & {"bottom_up_method_name": "OpenPose_BODY25B"}).fetch1(
                "keypoints"
            )
        else:
            raise Exception("Method not implemented")

        self.insert1(key)

    @staticmethod
    def joint_names(method="MMPose"):
        if method == "OpenPose":
            return OpenPosePerson.joint_names()
        elif method == "OpenPose_BODY25B":
            return [
                "Nose",
                "Left Eye",
                "Right Eye",
                "Left Ear",
                "Right Ear",
                "Left Shoulder",
                "Right Shoulder",
                "Left Elbow",
                "Right Elbow",
                "Left Wrist",
                "Right Wrist",
                "Left Hip",
                "Right Hip",
                "Left Knee",
                "Right Knee",
                "Left Ankle",
                "Right Ankle",
                "Neck",
                "Head",
                "Left Big Toe",
                "Left Little Toe",
                "Left Heel",
                "Right Big Toe",
                "Right Little Toe",
                "Right Heel",
            ]
        else:
            from .wrappers.mmpose import mmpose_joint_dictionary

            return mmpose_joint_dictionary[method]


@schema
class SkeletonAction(dj.Computed):
    definition = """
    -> TopDownPerson
    method            : varchar(50)
    ---
    top5              : longblob
    action_scores     : longblob
    label_map         : longblob
    action_window_len : int
    stride            : int
    computed_timestamp=CURRENT_TIMESTAMP : timestamp    # automatic timestamp
    """

    # Note: this will likely be refactored with a lookup table in the near future
    # to support different methods
    def make(self, key):

        from pose_pipeline.wrappers.mmaction import mmaction_skeleton_action_person

        key = mmaction_skeleton_action_person(key, stride=1)
        key["method"] = "mmaction_skeleton"
        self.insert1(key)


@schema
class SkeletonActionVideo(dj.Computed):
    definition = """
    -> SkeletonAction
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """

    def make(self, key):
        from pose_pipeline.utils.visualization import video_overlay, draw_keypoints

        video = (BlurredVideo & key).fetch1("output_video")
        keypoints = (TopDownPerson & key).fetch1("keypoints")

        bbox_fn = PersonBbox.get_overlay_fn(key)
        bbox = (PersonBbox & key).fetch1("bbox")

        top5_actions, stride = (SkeletonAction & key).fetch1("top5", "stride")

        def overlay_fn(image, idx):
            image = draw_keypoints(image, keypoints[idx], radius=20, color=(0, 255, 0))
            image = bbox_fn(image, idx, width=14, color=(0, 0, 255))

            if np.any(np.isnan(bbox[idx])):
                return image

            top5 = top5_actions[min(len(top5_actions) - 1, idx // stride)]

            top_left = bbox[idx, :2]
            for i, (action, score) in enumerate(top5):
                if score > 0.1:
                    label = f"{action.capitalize()}: {score:0.3}"

                    fontsize = 1.0e-3 * image.shape[0]
                    textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontsize, 4)[0]

                    coord = (int(top_left[0] + 5), int(top_left[1] + (10 + textsize[1]) * (1 + i)))
                    cv2.putText(image, label, coord, 0, fontsize, (255, 255, 255), thickness=4)

            return image

        fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        video_overlay(video, out_file_name, overlay_fn, downsample=1)

        key["output_video"] = out_file_name

        self.insert1(key)

        os.remove(out_file_name)
        os.remove(video)

        return out_file_name


@schema
class LiftingMethodLookup(dj.Lookup):
    definition = """
    lifting_method      : int
    ---
    lifting_method_name : varchar(50)
    """
    contents = [
        {"lifting_method": 0, "lifting_method_name": "GastNet"},
        {"lifting_method": 1, "lifting_method_name": "VideoPose3D"},
        {"lifting_method": 2, "lifting_method_name": "PoseAug"},
    ]


@schema
class LiftingMethod(dj.Manual):
    definition = """
    -> TopDownPerson
    -> LiftingMethodLookup
    """


@schema
class LiftingPerson(dj.Computed):
    definition = """
    -> LiftingMethod
    ---
    keypoints_3d       : longblob
    keypoints_valid    : longblob
    """

    def make(self, key):

        if (LiftingMethodLookup & key).fetch1("lifting_method_name") == "RIE":
            from .wrappers.rie_lifting import process_rie

            results = process_rie(key)
        elif (LiftingMethodLookup & key).fetch1("lifting_method_name") == "GastNet":
            from .wrappers.gastnet_lifting import process_gastnet

            results = process_gastnet(key)
        elif (LiftingMethodLookup & key).fetch1("lifting_method_name") == "VideoPose3D":
            from .wrappers.videopose3d import process_videopose3d

            results = process_videopose3d(key)
        elif (LiftingMethodLookup & key).fetch1("lifting_method_name") == "PoseAug":
            from .wrappers.poseaug import process_poseaug

            results = process_poseaug(key)
        else:
            raise Exception(f"Method not implemented {key}")

        key.update(results)
        self.insert1(key)

    def joint_names():
        """Lifting layers use Human3.6 ordering"""
        return [
            "Hip (root)",
            "Right hip",
            "Right knee",
            "Right foot",
            "Left hip",
            "Left knee",
            "Left foot",
            "Spine",
            "Thorax",
            "Nose",
            "Head",
            "Left shoulder",
            "Left elbow",
            "Left wrist",
            "Right shoulder",
            "Right elbow",
            "Right wrist",
        ]


@schema
class LiftingPersonVideo(dj.Computed):
    definition = """
    -> LiftingPerson
    -> BlurredVideo
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """

    def make(self, key):

        keypoints = (TopDownPerson & key).fetch1("keypoints")
        keypoints_3d = (LiftingPerson & key).fetch1("keypoints_3d").copy()
        blurred_video = (BlurredVideo & key).fetch1("output_video")
        width, height, fps = (VideoInfo & key).fetch1("width", "height", "fps")
        fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        with add_path(os.environ["GAST_PATH"]):

            from common.graph_utils import adj_mx_from_skeleton
            from common.skeleton import Skeleton
            from tools.inference import gen_pose
            from tools.preprocess import h36m_coco_format, revise_kpts

            from tools.vis_h36m import render_animation

            skeleton = Skeleton(
                parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31],
            )
            adj = adj_mx_from_skeleton(skeleton)

            joints_left, joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
            kps_left, kps_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
            rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
            keypoints_metadata = {
                "keypoints_symmetry": (joints_left, joints_right),
                "layout_name": "Human3.6M",
                "num_joints": 17,
            }

            keypoints_reformat, keypoints_score = keypoints[None, ..., :2], keypoints[None, ..., 2]
            keypoints, scores, valid_frames = h36m_coco_format(keypoints_reformat, keypoints_score)
            re_kpts = revise_kpts(keypoints, scores, valid_frames)
            re_kpts = re_kpts.transpose(1, 0, 2, 3)

            keypoints_3d[:, :, 2] -= np.amin(keypoints_3d[:, :, 2])
            anim_output = {"Reconstruction 1": keypoints_3d}

            render_animation(
                re_kpts,
                keypoints_metadata,
                anim_output,
                skeleton,
                fps,
                30000,
                np.array(70.0, dtype=np.float32),
                out_file_name,
                input_video_path=blurred_video,
                viewport=(width, height),
                com_reconstrcution=False,
            )

        key["output_video"] = out_file_name
        self.insert1(key)

        os.remove(blurred_video)
        os.remove(out_file_name)


## Classes that handle SMPL meshed based tracking
@schema
class SMPLMethodLookup(dj.Lookup):
    definition = """
    smpl_method       : int
    ---
    smpl_method_name  : varchar(50)
    """
    contents = [
        {"smpl_method": 0, "smpl_method_name": "VIBE"},
        {"smpl_method": 1, "smpl_method_name": "MEVA"},
        {"smpl_method": 2, "smpl_method_name": "ProHMR"},
        {"smpl_method": 3, "smpl_method_name": "Expose"},
        {"smpl_method": 4, "smpl_method_name": "PARE"},
        {"smpl_method": 5, "smpl_method_name": "PIXIE"},
        {"smpl_method": 6, "smpl_method_name": "ProHMR_MMPose"},
        {"smpl_method": 7, "smpl_method_name": "HybrIK"},
    ]


@schema
class SMPLMethod(dj.Manual):
    definition = """
    -> PersonBbox
    -> SMPLMethodLookup
    """


@schema
class SMPLPerson(dj.Computed):
    definition = """
    -> SMPLMethod
    ---
    model_type      : varchar(50)
    cams            : longblob
    poses           : longblob
    betas           : longblob
    joints3d        : longblob
    joints2d        : longblob
    """

    # verts           : longblob

    def make(self, key):

        smpl_method_name = (SMPLMethodLookup & key).fetch1("smpl_method_name")
        if smpl_method_name == "VIBE":

            from .wrappers.vibe import process_vibe

            res = process_vibe(key)
            res["model_type"] = "SMPL"

        elif smpl_method_name == "MEVA":

            from .wrappers.meva import process_meva

            res = process_meva(key)
            res["model_type"] = "SMPL"

        elif smpl_method_name == "ProHMR":

            from .wrappers.prohmr import process_prohmr

            res = process_prohmr(key)
            res["model_type"] = "SMPL"

        elif smpl_method_name == "ProHMR_MMPose":
            from .wrappers.prohmr import process_prohmr_mmpose

            res = process_prohmr_mmpose(key)
            res["model_type"] = "SMPL"

        elif smpl_method_name == "Expose":

            from .wrappers.expose import process_expose

            res = process_expose(key)
            res["model_type"] = "SMPL-X"

        elif smpl_method_name == "PARE":

            from .wrappers.pare import process_pare

            res = process_pare(key)
            res["model_type"] = "SMPL"

        elif smpl_method_name == "PIXIE":

            from .wrappers.pixie import process_pixie

            res = process_pixie(key)
            res["model_type"] = "SMPL-X"

        elif smpl_method_name == "HybrIK":

            from .wrappers.hybrik import process_hybrik

            res = process_hybrik(key)
            res["model_type"] = "SMPL"

        else:
            raise Exception(f"Method {smpl_method_name} not implemented")

        if "verts" in res.keys():
            res.pop("verts")

        self.insert1(res)

    @staticmethod
    def joint_names(model="smpl"):
        if model.upper() == "SMPL":
            from .utils.smpl import JOINT_NAMES_49

            return JOINT_NAMES_49
        elif model.upper() in ["SMPLX", "SMPL-X"]:
            from smplx.joint_names import JOINT_NAMES

            return JOINT_NAMES
        elif model.upper() == "PIXIE":
            # frustratingly, Pixie does not use the default keypoint ordering
            # TODO: can likely remove the cfg.model.extra_joint_path setting and get defaults
            with add_path(os.environ["PIXIE_PATH"]):
                from pixielib.models.SMPLX import SMPLX_names as pixie_joint_names
            return pixie_joint_names

    @staticmethod
    def smpl_joint_names(model="smpl"):
        from smplx.joint_names import JOINT_NAMES

        if model == "smpl":
            return JOINT_NAMES[:19]
        elif model == "smplx":
            # in smplx models the pelvis orientation is in a different field (global orientation)
            # as are the wrists
            return JOINT_NAMES[1:21]
        elif model == "PIXIE":
            # in addition to the dropped fields for smplx, Pixie also splits out the head and neck
            # into additional fields
            return [j for j in JOINT_NAMES[:20] if j not in ["pelvis", "head", "neck"]]
        else:
            raise Exception("Unknown model type")


@schema
class SMPLPersonVideo(dj.Computed):
    definition = """
    -> SMPLPerson
    -> BlurredVideo
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """

    def make(self, key):

        from pose_pipeline.utils.visualization import video_overlay

        poses, betas, cams = (SMPLPerson & key).fetch1("poses", "betas", "cams")

        smpl_method_name = (SMPLMethodLookup & key).fetch1("smpl_method_name")
        if smpl_method_name == "ProHMR" or smpl_method_name == "ProHMR_MMPose":
            from .wrappers.prohmr import get_prohmr_smpl_callback

            callback = get_prohmr_smpl_callback(key, poses, betas, cams)
        elif smpl_method_name == "Expose":
            from .wrappers.expose import get_expose_callback

            callback = get_expose_callback(key)
        elif smpl_method_name == "PIXIE":
            from .wrappers.pixie import get_pixie_callback

            callback = get_pixie_callback(key)

        elif smpl_method_name == "HybrIK":
            from .wrappers.hybrik import get_hybrik_smpl_callback

            callback = get_hybrik_smpl_callback(key)

        else:
            from pose_pipeline.utils.visualization import get_smpl_callback

            callback = get_smpl_callback(key, poses, betas, cams)

        video = (BlurredVideo & key).fetch1("output_video")

        fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        video_overlay(video, out_file_name, callback, downsample=1)
        key["output_video"] = out_file_name

        self.insert1(key)

        os.remove(video)


@schema
class CenterHMR(dj.Computed):
    definition = """
    -> Video
    ---
    results           : longblob
    """

    def make(self, key):

        with add_path(
            [os.path.join(os.environ["CENTERHMR_PATH"], "src"), os.path.join(os.environ["CENTERHMR_PATH"], "src/core")]
        ):
            from pose_pipeline.wrappers.centerhmr import centerhmr_parse_video

            video = Video.get_robust_reader(key, return_cap=False)
            res = centerhmr_parse_video(video, os.environ["CENTERHMR_PATH"])

        # don't store verticies or images
        keys_to_keep = ["params", "pj2d", "j3d", "j3d_smpl24", "j3d_spin24", "j3d_op25"]
        res = [{k: v for k, v in r.items() if k in keys_to_keep} for r in res]
        key["results"] = res

        self.insert1(key)

        # not saving the video in database, just to reduce space requirements
        os.remove(video)


@schema
class CenterHMRPerson(dj.Computed):
    definition = """
    -> PersonBbox
    -> CenterHMR
    -> VideoInfo
    ---
    keypoints        : longblob
    poses            : longblob
    betas            : longblob
    cams             : longblob
    global_orients   : longblob
    centerhmr_ids    : longblob
    """

    def make(self, key):

        width, height = (VideoInfo & key).fetch1("width", "height")

        def convert_keypoints_to_image(keypoints, imsize=[width, height]):
            mp = np.array(imsize) * 0.5
            scale = np.max(np.array(imsize)) * 0.5

            keypoints_image = keypoints * scale + mp
            return list(keypoints_image)

        # fetch data
        hmr_results = (CenterHMR & key).fetch1("results")
        bbox = (PersonBbox & key).fetch1("bbox")

        # get the 2D keypoints. note these are scaled from (-0.5, 0.5) assuming a
        # square image (hence convert_keypoints_to_image)
        pj2d = [r["pj2d"] if "pj2d" in r.keys() else np.zeros((0, 25, 2)) for r in hmr_results]
        all_matches = [
            match_keypoints_to_bbox(bbox[idx], convert_keypoints_to_image(pj2d[idx]), visible=False)
            for idx in range(bbox.shape[0])
        ]

        keypoints, centerhmr_ids = list(zip(*all_matches))

        key["poses"] = np.asarray(
            [
                res["params"]["body_pose"][id] if id is not None else np.array([np.nan] * 69) * np.nan
                for res, id in zip(hmr_results, centerhmr_ids)
            ]
        )
        key["betas"] = np.asarray(
            [
                res["params"]["betas"][id] if id is not None else np.array([np.nan] * 10) * np.nan
                for res, id in zip(hmr_results, centerhmr_ids)
            ]
        )
        key["cams"] = np.asarray(
            [
                res["params"]["cam"][id] if id is not None else np.array([np.nan] * 3) * np.nan
                for res, id in zip(hmr_results, centerhmr_ids)
            ]
        )
        key["global_orients"] = np.asarray(
            [
                res["params"]["global_orient"][id] if id is not None else np.array([np.nan] * 3) * np.nan
                for res, id in zip(hmr_results, centerhmr_ids)
            ]
        )

        key["keypoints"] = np.asarray(keypoints)
        key["centerhmr_ids"] = np.asarray(centerhmr_ids)

        self.insert1(key)

    @staticmethod
    def joint_names():
        from smplx.joint_names import JOINT_NAMES

        return JOINT_NAMES[:23]


@schema
class CenterHMRPersonVideo(dj.Computed):
    definition = """
    -> CenterHMRPerson
    -> BlurredVideo
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """

    def make(self, key):

        from pose_estimation.util.pyrender_renderer import PyrendererRenderer
        from pose_estimation.body_models.smpl import SMPL
        from pose_pipeline.utils.visualization import video_overlay

        # fetch data
        pose_data = (CenterHMRPerson & key).fetch1()
        video_filename = (BlurredVideo & key).fetch1("output_video")

        fd, fname = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        video = (BlurredVideo & key).fetch1("output_video")

        smpl = SMPL()

        def overlay(image, idx):
            body_pose = np.concatenate([pose_data["global_orients"][idx], pose_data["poses"][idx]])
            body_beta = pose_data["betas"][idx]

            if np.any(np.isnan(body_pose)):
                return image

            h, w = image.shape[:2]
            if overlay.renderer is None:
                overlay.renderer = PyrendererRenderer(smpl.get_faces(), (h, w))

            verts = smpl(body_pose.astype(np.float32)[None, ...], body_beta.astype(np.float32)[None, ...])[0][0]

            cam = [pose_data["cams"][idx][0], *pose_data["cams"][idx][:3]]
            if h > w:
                cam[0] = 1.1 ** cam[0] * (h / w)
                cam[1] = 1.1 ** cam[1]
            else:
                cam[0] = 1.1 ** cam[0]
                cam[1] = (1.1 ** cam[1]) * (w / h)

            return overlay.renderer(verts, cam, img=image)

        overlay.renderer = None

        ofd, out_file_name = tempfile.mkstemp(suffix=".mp4")
        os.close(ofd)
        video_overlay(video, out_file_name, overlay, downsample=4)
        key["output_video"] = out_file_name

        self.insert1(key)

        os.remove(out_file_name)
        os.remove(video)


@schema
class HumorPerson(dj.Computed):
    definition = """
    -> OpenPosePerson
    ----
    trans         : longblob
    root_orient   : longblob
    pose_body     : longblob
    betas         : longblob
    latent_pose   : longblob
    latent_motion : longblob
    floor_plane   : longblob
    contacts      : longblob
    vertices      : longblob
    faces         : longblob
    """

    def make(self, key):

        from pose_pipeline.wrappers.humor import process_humor

        res = process_humor(key)

        self.insert1(res)


@schema
class HumorPersonVideo(dj.Computed):
    definition = """
    -> HumorPerson
    ----
    output_video      : attach@localattach    # datajoint managed video file
    """

    def make(self, key):

        from pose_pipeline.wrappers.humor import render_humor

        video = render_humor(key)
        key["output_video"] = video

        self.insert1(key)


@schema
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

    @staticmethod
    def joint_names():
        """PoseFormer follows the output format of Video3D and uses Human3.6 ordering"""
        return [
            "Hip (root)",
            "Right hip",
            "Right knee",
            "Right foot",
            "Left hip",
            "Left knee",
            "Left foot",
            "Spine",
            "Thorax",
            "Nose",
            "Head",
            "Left shoulder",
            "Left elbow",
            "Left wrist",
            "Right shoulder",
            "Right elbow",
            "Right wrist",
        ]

@schema
class TrackingBboxQR(dj.Computed):
    definition = """
    -> Video
    ---
    total_frames           : int
    qr_detected_frames     : int
    qr_decoded_frames      : int
    qr_results             : longblob
    """

    def make(self, key):
        print(key)
        from pose_pipeline.utils.tracking import detect_qr_code
        import pose_pipeline.utils.tracking as tracking
        from tqdm import tqdm
        import matplotlib

        import subprocess
        import os
        def compress(video):
            fd, outfile = tempfile.mkstemp(suffix=".mp4")
            subprocess.run(["ffmpeg", "-y", "-i", video, "-c:v", "libx264","-loglevel", "warning", "-b:v", "1M", outfile])
            os.close(fd)
            return outfile

        # Fetch the video and tracks from the respective tables
        video = (Video & key).fetch1("video")
        # tracks = (TrackingBbox & key).fetch1("tracks")

        compressed_video = compress(video)

        # Create OpenCV video capture object to go through each frame
        cap = cv2.VideoCapture(compressed_video)

        # get video info (total frames, frame height/width, frames per second)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)

        downsample = 4
        visualize = True

        # configure output
        output_size = (int(w / downsample), int(h / downsample))


        import ipywidgets as widgets
        from IPython.display import display
        owidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/downsample)
        oheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/downsample)

        canvas = widgets.Image(width=owidth, height=oheight)
        container = widgets.HBox([canvas])

        display(container)

        # Setting up initial QR tracking info
        # This is the number of frames between QR detections before the search area will be expanded
        qr_missing_frames_allowed = 600
        qr_missing_cnt = 0
        detected_flag = False

        # This is the % to take off of each side to create the bbox enclosing the search area
        border_pct = 20
        print(f"h: {h}, w: {w}")

        x1 = int(w * border_pct * 0.01)
        y1 = int(h * border_pct * 0.01)
        x2 = int(w - w * border_pct * 0.01)
        y2 = int(h - h * border_pct * 0.01)

        bbox = np.array([x1,y1,x2,y2])

        print("setting up")
        unique_qr_decoded_text = []
        track_id_qr_detection = {}

        qr_bbox = []
        qr_decoding = []

        counts = {
            "detections": 0,
            "decoding": 0,
        }

        # Initializing QReader
        qr_reader = tracking.QReaderdetector()

        print("detecting qr")

        for idx in tqdm(range(total_frames)):

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            out_image = frame.copy()
            qr_image = frame.copy()

            frame_qr_bbox = []
            frame_qr_decoded_text = []

            image = qr_image.copy()
            small = int(5e-3 * np.max(image.shape))
            large = 2 * small
        

            # Run the QR detection method, which returns [decoded text, top left of QR code, bottom right of QR code] or False
            qr_detection = detect_qr_code(image, bbox, qr_reader)

            # This variable is False if no QR codes are detected
            if qr_detection != False:

                qr_missing_cnt = 0
                detected_flag = True

                # Unpacking the output of the QR detection method
                decodedText, top_left_tuple, bottom_right_tuple = qr_detection

                # Increment the count of total QR detections
                counts["detections"] += 1

                # Save the coordinates for the bounding box around the QR code
                frame_qr_bbox = [top_left_tuple, bottom_right_tuple]

                # Flatten the bbox coords
                bbox_coords = [coord for corner in frame_qr_bbox for coord in corner]
                
                # Get the new search area for QR codes
                qr_h = abs(bbox_coords[3] - bbox_coords[1])
                qr_w = abs(bbox_coords[2] - bbox_coords[0])
                
                new_search_coord_x1 = bbox_coords[0] - 2*qr_w 
                new_search_coord_y1 = bbox_coords[1] - 2*qr_h 
                new_search_coord_x2 = bbox_coords[2] + 2*qr_w
                new_search_coord_y2 = bbox_coords[3] + 2*qr_h

                # confirm you are going to stay in the bounds of the image
                if new_search_coord_x1 > 0 and new_search_coord_y1 > 0 and new_search_coord_x2 < w and new_search_coord_y2 < h: 
                    # A QR code was detected, now we can search in a smaller area around the bbox
                    # x1 = max(0,int(new_search_coord_x1))
                    # y1 = max(0,int(new_search_coord_y1))
                    # x2 = min(w,int(new_search_coord_x2))
                    # y2 = min(h,int(new_search_coord_y2))
                    x1 = int(new_search_coord_x1)
                    y1 = int(new_search_coord_y1)
                    x2 = int(new_search_coord_x2)
                    y2 = int(new_search_coord_y2)

                bbox = np.array([x1,y1,x2,y2])

                # draw a rectangle around the QR code based on the location passed from the detection method
                cv2.rectangle(out_image, top_left_tuple, bottom_right_tuple, color=(0, 0, 255), thickness=10)

                frame_qr_decoded_text = decodedText
                # frame_qr_decoded_text = [decodedText]

                # Increment the count of correct QR decoding
                if decodedText != "":
                    counts["decoding"] += 1

                # Keep track of the strings decoded during the video
                if decodedText not in unique_qr_decoded_text:
                    unique_qr_decoded_text.append(decodedText)

            else:
                qr_missing_cnt += 1 
        
                # If there have been no QR detections for the allowed # of frames, expand the search area
                if qr_missing_cnt == qr_missing_frames_allowed:
                    qr_missing_cnt = 0
                    # if there was already a detection, expand the area around the last QR detection
                    if detected_flag:
                        new_search_coord_x1 -= qr_w 
                        new_search_coord_y1 -= qr_h 
                        new_search_coord_x2 += qr_w
                        new_search_coord_y2 += qr_h
                        
                        # confirm you are going to stay in the bounds of the image
                        x1 = max(0,int(new_search_coord_x1))
                        y1 = max(0,int(new_search_coord_y1))
                        x2 = min(w,int(new_search_coord_x2))
                        y2 = min(h,int(new_search_coord_y2))

                        bbox = np.array([x1,y1,x2,y2])
                            
                    else:
                        # otherwise expand the original bbox size
                        x1 = int(w * border_pct * 0.5 * 0.01)
                        y1 = int(h * border_pct * 0.5 * 0.01)
                        x2 = int(w - w * border_pct * 0.5 * 0.01)
                        y2 = int(h - h * border_pct * 0.5 * 0.01)

                        bbox = np.array([x1,y1,x2,y2])
                        
            # Draw the bounding box for the current track ID for the current frame
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), large)
            cv2.rectangle(out_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), small)

            # Add the coordinates for the current qr_bbox to the list for all frames
            qr_bbox.append(frame_qr_bbox)
            # Add the decoded text to the list for all frames
            qr_decoding.append(frame_qr_decoded_text)

            # move back to BGR format and write to movie
            out_frame = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
            out_frame = cv2.resize(out_image, output_size)

            if visualize:
                #out_frame = cv2.resize(out_frame, (0,0), fx=1/downsample, fy=1/downsample)
                ret, jpeg = cv2.imencode('.jpg', out_frame)
                canvas.value = jpeg.tobytes()
                #cv2.imshow("image", out_frame)
                #cv2.waitKey(1)

        print(f"Detected the QR code in {counts['detections']} out of {total_frames} frames ({float(counts['detections'] / total_frames)}).")
        print(f"Text Decoded: {unique_qr_decoded_text} {counts['decoding']} times")
        print("TRACK IDS WITH QR DETECTIONS")

        track_id_qr_detection["qr_bbox"] = qr_bbox
        track_id_qr_detection["qr_decoded"] = unique_qr_decoded_text
        # If there is anything besides [], then there was a detection in that frame
        track_id_qr_detection["qr_info_by_frame"] = qr_decoding


        # Save the QR information for the current video
        key["total_frames"]       = total_frames
        key["qr_detected_frames"] = counts['detections']
        key["qr_decoded_frames"]  = counts['decoding']
        key["qr_results"]         = track_id_qr_detection
        self.insert1(key)

        cv2.destroyAllWindows()
        cap.release()

@schema
class TrackingBboxQRByID(dj.Computed):
    definition = """
    -> TrackingBbox
    ---
    qr_results_by_id     : longblob
    """

    def make(self, key):
        print(key)
        from tqdm import tqdm
        import matplotlib

        import subprocess
        import os

        visualize = False

        if visualize:
            def compress(video):
                fd, outfile = tempfile.mkstemp(suffix=".mp4")
                subprocess.run(["ffmpeg", "-y", "-i", video, "-c:v", "libx264","-loglevel", "warning", "-b:v", "1M", outfile])
                os.close(fd)
                return outfile
            # Fetch the video and tracks from the respective tables
            video = (Video & key).fetch1("video")

            compressed_video = compress(video)

            # Create OpenCV video capture object to go through each frame
            cap = cv2.VideoCapture(compressed_video)

            # get video info (total frames, frame height/width, frames per second)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = cap.get(cv2.CAP_PROP_FPS)

            downsample = 4
            visualize = True

            # configure output
            output_size = (int(w / downsample), int(h / downsample))

            # Get the number of unique track IDs and generate colors for each
            N = len(np.unique([t["track_id"] for track in tracks for t in track]))
            colors = matplotlib.cm.get_cmap("hsv", lut=N)


            import ipywidgets as widgets
            from IPython.display import display
            owidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/downsample)
            oheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/downsample)

            canvas = widgets.Image(width=owidth, height=oheight)
            container = widgets.HBox([canvas])

            display(container)

        def compute_intersection_area(qr_bbox,track_bbox):
            # Get the top left and bottom right of the intersection of the qr code and the track
            top_left = np.maximum(qr_bbox[:2],track_bbox[:2])
            bottom_right = np.minimum(qr_bbox[2:],track_bbox[2:])
            
            # if there is no intersection then exit
            if np.any(top_left > bottom_right):
                return 0.
            
            # Get dimensions and area of intersection
            intersection_dims = bottom_right - top_left
            intersection_area = np.prod(intersection_dims) 
            
            return intersection_area


        # Fetch the tracks
        tracks = (TrackingBbox & key).fetch1("tracks")

        qr_results,total_frames = (TrackingBboxQR & key).fetch1("qr_results","total_frames")

        frames = min(total_frames,len(tracks))

        # Unpack results from QR detection
        qr_bboxes    = qr_results['qr_bbox']

        qr_info_list = []

        for idx in tqdm(range(frames)):

            #######################################################################################################################
            if visualize:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                out_image = frame.copy()
                qr_image = frame.copy()
            #######################################################################################################################

            detected_info_dict = {}
            
            # Get the bbox for the current frame if it exists
            current_bbox = qr_bboxes[idx]

            if current_bbox != []:
                # This means there was a detection in the current frame
                
                # Make the bbox easier to work with
                # Flatten and convert to numpy array
                qr_bbox = np.array([coord for corner in current_bbox for coord in corner])
                
                # Calculate the area of the qr bbox
                qr_area = np.prod(qr_bbox[2:] - qr_bbox[:2])
                
                # Confirm the qr bbox is not empty
                if qr_area > 0:
                    if visualize:
                        cv2.rectangle(out_image, current_bbox[0], current_bbox[1], color=(0, 0, 255), thickness=10)
                    # Cycle through each track in the current frame
                    for track in tracks[idx]:

                        # Getting the track id and bounding box from the current track
                        track_id = track["track_id"]
                        bbox = track["tlbr"]

                        # Check if the qr bbox is contained in the track
                        intersection_area = compute_intersection_area(qr_bbox,bbox)

                        # calculate percentage of intersection
                        intersection_pct = intersection_area/qr_area
                        
                        detected_info_dict[track_id] = intersection_pct

                        #######################################################################################################################
                        if visualize:
                            c = colors(track_id)
                            c = (int(c[0] * 255.0), int(c[1] * 255.0), int(c[2] * 255.0))

                            # Making a copy of the current frame and determining sizes for bounding boxes
                            image = qr_image.copy()
                            small = int(5e-3 * np.max(image.shape))
                            large = 2 * small

                            # Draw the bounding box for the current track ID for the current frame
                            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), large)
                            cv2.rectangle(out_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), c, small)

                            # Add track ID text to the image
                            label = str(track_id)
                            textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, int(5.0e-3 * out_image.shape[0]), 4)[0]
                            x = int((bbox[0] + bbox[2]) / 2 - textsize[0] / 2)
                            y = int((bbox[3] + bbox[1]) / 2 + textsize[1] / 2)
                            cv2.putText(out_image, label, (x, y), 0, 5.0e-3 * out_image.shape[0], (255, 255, 255), thickness=large)
                            cv2.putText(out_image, label, (x, y), 0, 5.0e-3 * out_image.shape[0], c, thickness=small)
                        #######################################################################################################################

            # Append the detected info to the list for all frames
            qr_info_list.append(detected_info_dict)


            #######################################################################################################################
            if visualize:
                # move back to BGR format and write to movie
                out_frame = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
                out_frame = cv2.resize(out_image, output_size)

                #out_frame = cv2.resize(out_frame, (0,0), fx=1/downsample, fy=1/downsample)
                ret, jpeg = cv2.imencode('.jpg', out_frame)
                canvas.value = jpeg.tobytes()
                #cv2.imshow("image", out_frame)
                #cv2.waitKey(1)
            #######################################################################################################################

        # Save the QR information for the current video
        key["qr_results_by_id"] = qr_info_list
        self.insert1(key)
        #######################################################################################################################
        if visualize:
            cv2.destroyAllWindows()
            cap.release()
        #######################################################################################################################

@schema
class TrackingBboxQRWindowSelect(dj.Manual):
    definition = """
    -> TrackingBboxQR
    window_len              : int
    """

@schema
class TrackingBboxQRMetrics(dj.Computed):
    definition = """
    -> TrackingBboxQRWindowSelect
    ---
    total_frames                  : int
    likely_tracks                 : longblob
    qr_detected_frames            : int
    qr_decoded_frames             : int
    likely_id_overlap             : longblob
    participant_frame_count       : int
    qr_calculated_frame_metrics   : longblob
    consecutive_frames_by_id      : longblob
    """

    def make(self, key):
        # Key will have video_project, filename, tracking_method AND window_len
        # window_len is size of sliding window used to calculate the likely ids
        print(key)
        from pose_pipeline.utils.tracking_evaluation import compute_temporal_overlap, process_detections, process_decodings, get_likely_ids, get_participant_frame_count, get_unique_ids, get_ids_in_frame, compute_consecutive_frames

        window_len = key['window_len']
        qr_calculated_frame_metrics = {}

        # Get qr data for current video
        tracking_method, qr_results = (TrackingBboxQR & key).fetch1('tracking_method','qr_results')
        # Get tracks data for current video
        tracks, num_tracks = (TrackingBbox & key ).fetch1('tracks','num_tracks')

        # Get the unique track IDs that appear in the current video
        all_track_ids = get_ids_in_frame(tracks)
        unique_ids = get_unique_ids(all_track_ids)

        # Get the consecutive frame lists
        consecutive_frame_list = compute_consecutive_frames(unique_ids, all_track_ids)

        # Extract frame QR data for current video
        frame_data_tmp = qr_results['frame_data_dict']
        frame_data = pd.DataFrame(frame_data_tmp)

        total_frames = len(tracks)

        # Calculate frame overlap and counts for each track ID based on tracks data
        overlaps, track_id_counts = compute_temporal_overlap(tracks,unique_ids)

        if qr_results['qr_counts']['detections'] > 1:

            # Get the number of detections and decodings
            detection_by_frame = process_detections(frame_data)
            decoding_by_frame = process_decodings(frame_data) 

            qr_detections = len(detection_by_frame)
            qr_decodings = len(decoding_by_frame[(decoding_by_frame.T != 0).any()])

            print("Total frames: ",total_frames)
            print(f"Frames with detections: {qr_detections} ({qr_detections/total_frames})")
            print(f"Frames with decoding: {qr_decodings} ({qr_decodings/total_frames})")

        
            # If the window length is larger than the number of detections then use 1 instead
            if len(detection_by_frame) < window_len:
                window_len = 1

        
            # Find IDs that are most likely to correspond to the participant 
            likely_ids, likely_ids_df, all_detected_ids, all_decoded_ids = get_likely_ids(detection_by_frame, decoding_by_frame,consecutive_frame_list,all_track_ids, window_len)

            qr_calculated_frame_metrics['likely_ids_by_frame'] = likely_ids_df['likely_ids'].values
            qr_calculated_frame_metrics['ids_with_det_by_frame'] = likely_ids_df['tentative_likely_ids'].values
            qr_calculated_frame_metrics['frame_idx_with_det'] = np.array(likely_ids_df.index)
            qr_calculated_frame_metrics['detection_by_frame'] = detection_by_frame.values
            qr_calculated_frame_metrics['decoding_by_frame'] = decoding_by_frame.values


            print(f"Likely Participant ID(s): {likely_ids}")

            # Check if the likely IDs appear in the frame together. If they do then probably ID swap, otherwise, relabeling
            
            temporal_overlap = overlaps.loc[likely_ids,likely_ids]

            for i in range(len(likely_ids)):
                for j in range(i+1, len(likely_ids)): 
                    likely_id_a = likely_ids[i]
                    likely_id_b = likely_ids[j]
                    print(f"Temporal overlap between track IDs {likely_id_a} and {likely_id_b}: {temporal_overlap.loc[likely_id_a,likely_id_b]} frames")

            likely_id_overlap = temporal_overlap.values

            # Check how many frames the likely IDs appeared in the video (based on the tracking algo)
            participant_in_frame = get_participant_frame_count(tracks,likely_ids)

            print(f"Frames with participant (from tracks): {participant_in_frame} ({participant_in_frame/total_frames})")

        else:
            likely_ids = []
            qr_calculated_frame_metrics = {}
            likely_id_overlap = []
            participant_in_frame = 0

            qr_detections = 0
            qr_decodings = 0

            print("Total frames: ",total_frames)
            print(f"Frames with detections: {qr_detections} ({qr_detections/total_frames})")
            print(f"Frames with decoding: {qr_decodings} ({qr_decodings/total_frames})")

        # Save the QR information for the current video
        key["total_frames"]            = total_frames
        key["likely_tracks"]           = likely_ids
        key["qr_detected_frames"]      = qr_detections
        key["qr_decoded_frames"]       = qr_decodings
        key["likely_id_overlap"]       = likely_id_overlap
        key["participant_frame_count"] = participant_in_frame
        key["qr_calculated_frame_metrics"] = qr_calculated_frame_metrics
        key["consecutive_frames_by_id"] = consecutive_frame_list

        self.insert1(key)

@schema
class TrackingBboxSplitSelect(dj.Manual):
    definition = """
    -> TrackingBboxQRMetrics
    missing_frame_threshold : int
    """


@schema
class TrackingBboxSplits(dj.Computed):
    definition = """
    -> TrackingBboxSplitSelect
    ---
    total_split_sum         : int
    total_split_frequency   : float
    splits_by_id            : longblob
    splits_frequency        : longblob
    consecutive_frames      : longblob
    """

    def make(self, key):
        # Key will have video_project, filename, tracking_method, and missing_frames_threshold
        # missing_frames_threshold is number of frames that a track can be missing from a video before it counts as a split
        print(key)
        from pose_pipeline.utils.tracking_evaluation import get_unique_ids, get_ids_in_frame

        missing_frame_threshold = key['missing_frame_threshold']

        # Get tracks data for current video
        tracks = (TrackingBbox & key ).fetch1('tracks')
        likely_tracks, consecutive_frames_by_id = (TrackingBboxQRMetrics & key).fetch1('likely_tracks','consecutive_frames_by_id')

        # Get the unique track IDs that appear in the current video
        all_track_ids = get_ids_in_frame(tracks)
        unique_ids = get_unique_ids(all_track_ids)

        # # Get the splits and consecutive frame lists
        # splits, consecutive_frame_list = compute_splits(unique_ids, all_track_ids, missing_frame_threshold)

        splits = {}
        for track_id in consecutive_frames_by_id:
            # go through the list of consecutive frames for each ID
            splits[track_id] = 0
            for f in range(1,len(consecutive_frames_by_id[track_id])):
                gap = consecutive_frames_by_id[track_id][f][0] - consecutive_frames_by_id[track_id][f-1][1] - 1

                if gap > missing_frame_threshold:
                    splits[track_id] += 1

        # Get split sum and frequency for likely tracks
        likely_track_splits = [splits[s] for s in likely_tracks]
        total_split_sum = sum(likely_track_splits)

        # To calculate frequency, get overall splits per minute for the likely tracks ((total_splits/total_frames)*fps*60)
        total_split_frequency = float(total_split_sum)/len(tracks) * 30 * 60

        # Calculating splits per minute by ID
        split_frequency = {}
        for id in splits:
            # Calculate the total number of frames the id appeared
            id_frame_total = 0.
            for frames in consecutive_frames_by_id[id]:
                # Summing each consecutive window for each ID (+1 since frames saved are inclusive ranges)
                id_frame_total += (frames[1] - frames[0]) + 1

            split_frequency[id] = (splits[id]/id_frame_total) * 30 * 60

        
        # Save the splits information for the current video
        key["total_split_sum"]          = total_split_sum
        key["total_split_frequency"]    = total_split_frequency
        key["splits_by_id"]             = splits
        key["splits_frequency"]          = split_frequency
        key["consecutive_frames"]       = consecutive_frames_by_id

        self.insert1(key)


@schema
class TrackingBboxIOUThreshold(dj.Manual):
    definition = """
    -> TrackingBboxQRMetrics
    iou_threshold : decimal(5,5)
    """


@schema
class TrackingBboxSwaps(dj.Computed):
    definition = """
    -> TrackingBboxIOUThreshold
    ---
    total_id_swaps              : int
    id_swap_frequency           : float
    total_spatial_overlaps      : int
    spatial_overlap_frequency   : float
    total_relabels              : int
    relabel_frequency           : float
    """

    def make(self, key):
        # Key will have video_project, filename, tracking_method, window_len, and iou_threshold
        # iou_threshold is the IOU value below which is considered an ID swap. Above is just a spatial overlap
        print(key)
        from pose_pipeline.utils.tracking_evaluation import get_ids_in_frame, get_bboxes_in_frame, determine_id_swaps

        iou_threshold = key['iou_threshold']

        # Get tracks data for current video
        tracks = (TrackingBbox & key ).fetch1('tracks')
        # Get calculated metrics
        qr_calculated_frame_metrics = (TrackingBboxQRMetrics & key).fetch1('qr_calculated_frame_metrics')

        if qr_calculated_frame_metrics != {}:
            likely_ids_by_frame    = qr_calculated_frame_metrics['likely_ids_by_frame']
            ids_with_det_by_frame  = qr_calculated_frame_metrics['ids_with_det_by_frame']
            frame_idx_with_det     = qr_calculated_frame_metrics['frame_idx_with_det']
            detection_by_frame     = qr_calculated_frame_metrics['detection_by_frame']

            # Get the unique track IDs that appear in the current video
            all_track_ids = get_ids_in_frame(tracks)

            # Get all bboxes that appear in the video
            bboxes_in_frame = get_bboxes_in_frame(tracks)

            iou_array, spatial_overlap, id_swap, relabeling = determine_id_swaps(detection_by_frame, likely_ids_by_frame, all_track_ids, frame_idx_with_det, ids_with_det_by_frame, bboxes_in_frame, iou_threshold)
            
            spatial_overlap_sum = sum(spatial_overlap)
            id_swap_sum = sum(id_swap)
            relabeling_sum = sum(relabeling)

            spatial_overlap_freq = spatial_overlap_sum/len(tracks) * 30. * 60.
            id_swap_freq = id_swap_sum/len(tracks) * 30. * 60.
            relabeling_freq = relabeling_sum/len(tracks) * 30. * 60.

            print("Relabeling: ",relabeling_sum,relabeling_freq)
            print("ID Swap: ",id_swap_sum,id_swap_freq)
            print("Spatial Overlap: ",spatial_overlap_sum,spatial_overlap_freq)


            # Save the id swap information for the current video
            key["total_id_swaps"]            = id_swap_sum
            key["id_swap_frequency"]         = id_swap_freq
            key["total_spatial_overlaps"]    = spatial_overlap_sum
            key["spatial_overlap_frequency"] = spatial_overlap_freq
            key["total_relabels"]            = relabeling_sum
            key["relabel_frequency"]         = relabeling_freq

            self.insert1(key)