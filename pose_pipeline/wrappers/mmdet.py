import os
import cv2
import numpy as np
from tqdm import tqdm
import mmdet.apis
from mmdet.utils import register_all_modules
from mim import download 

package = 'mmdet'


def mmdet_bounding_boxes(file_path, method="deepsort"):

    from pose_pipeline import MODEL_DATA_DIR

    if method == "deepsort":
        
        # Define the model config id and checkpoints
        deepsort_config_id = "deepsort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval"
        detector_checkpoint_name = "faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth"
        reid_checkpoint_name = "tracktor_reid_r50_iter25245-a452f51f.pth"

        # define the destination folder
        destination = os.path.join(MODEL_DATA_DIR, f"mmdetection/{method}/")

        # download the model and checkpoints
        download(package, [deepsort_config_id], dest_root=destination)

        # define the model config and checkpoints paths
        model_config = os.path.join(destination, f"{deepsort_config_id}.py")
        detector_checkpoint = os.path.join(destination, detector_checkpoint_name)
        reid_checkpoint = os.path.join(destination, reid_checkpoint_name)

        # register all modules from mmdet
        register_all_modules()

    elif method == "qdtrack":

        # Define the model config id and checkpoints
        qdtrack_config_id = "qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval"
        qdtrack_checkpoint_name = "qdtrack_faster-rcnn_r50_fpn_4e_mot17_20220315_145635-76f295ef.pth"

        # define the destination folder
        destination = os.path.join(MODEL_DATA_DIR, f"mmdetection/{method}/")

        # download the model and checkpoints
        download(package, [qdtrack_config_id], dest_root=destination)

        # define the model config and checkpoints paths
        model_config = os.path.join(destination, f"{qdtrack_config_id}.py")
        detector_checkpoint = os.path.join(destination, qdtrack_checkpoint_name)
        reid_checkpoint = None

        # register all modules from mmdet
        register_all_modules()
    else:
        raise Exception(f"Unknown config file for MMDetection method {method}")
    
    # initialize the model
    model = mmdet.apis.init_track_model(config=model_config, detector=detector_checkpoint, reid=reid_checkpoint, device="cpu")

    cap = cv2.VideoCapture(file_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracks = []

    for frame_id in tqdm(range(video_length)):
        ret, frame = cap.read()

        if ret != True or frame is None:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = mmdet.apis.inference_mot(model=model, img=frame, frame_id=frame_id, video_len=video_length)

        # MMDet uses a custom data structure to store the results
        bboxes = result.video_data_samples[0].pred_track_instances.bboxes.numpy()
        track_ids = result.video_data_samples[0].pred_track_instances.instances_id
        confidences = result.video_data_samples[0].pred_track_instances.scores

        assert len(bboxes) == len(track_ids) == len(confidences)

        result_len = len(bboxes)

        tracks.append(
            [
                {
                    "track_id": int(track_ids[i].item()),
                    "tlbr": bboxes[i],
                    "tlhw": np.array([bboxes[i][0], bboxes[i][1], bboxes[i][2] - bboxes[i][0], bboxes[i][3] - bboxes[i][1]]),
                    "confidence": confidences[i].item(),
                }
                for i in range(result_len)
            ]
        )

    return tracks
