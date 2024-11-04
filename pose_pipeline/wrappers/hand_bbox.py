import os
import cv2
import numpy as np
import datajoint as dj
from pose_pipeline import Video


def mmpose_hand_det(key, method='RTMDet'):

    from mmpose.apis import init_model
    try:
        from mmdet.apis import inference_detector, init_detector
        has_mmdet =True
    except (ImportError, ModuleNotFoundError):
        has_mmdet = False 
    from mmpose.utils import adapt_mmdet_pipeline
    from mmpose.evaluation.functional import nms

    video =  Video.get_robust_reader(key, return_cap=False) # returning video allows deleting it
    path = os.path.dirname(os.path.abspath(__file__))

    if method == 'RTMDet':
        detection_cfg = os.path.join(path,'../../3rdparty/mmpose/config/hand_2d_keypoint/rtmdet_nano_320-8xb32_hand.py')
        detection_ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth'
        device = 'cpu'

    # build detector
    detector = init_detector(detection_cfg, detection_ckpt, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    
    #capture video
    cap = cv2.VideoCapture(video)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    boxes_list = []
    num_boxes = 0
    # iterate trough frames
    for frame_id in range(video_length):
        ret, frame = cap.read()
        assert ret and frame is not None
        #get detection results
        det_result = inference_detector(detector, frame)
        pred_instance = det_result.pred_instances.cpu().numpy()

        #calculate bboxes confidences
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        #capture bboxes with higher than 0.3 score
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                    pred_instance.scores > .3)]
        #overlap highest scoring boxes to get cohesive boxes
        bboxes = bboxes[nms(bboxes, .3), :4]
        #expand bboxes by 100 pixels
        bboxes[:,:2] -= 100
        bboxes[:,-2:] += 100
        if(bboxes.shape[0] > num_boxes):
            num_boxes = bboxes.shape[0]
        boxes_list.append(bboxes)

    cap.release()
    os.remove(video)

    return num_boxes, boxes_list


def extract_xy_min_max(points, width, height):
    # for point in points:
    xmin = points[:,0] - width / 2
    xmin_min = np.min(xmin)

    xmax = points[:,0] + width / 2
    xmax_max = np.max(xmax)

    ymin = points[:,1] - height / 2
    ymin_min = np.min(ymin)

    ymax = points[:,1] + height / 2
    ymax_max = np.max(ymax)

    return np.asarray([xmin_min, ymin_min, xmax_max, ymax_max])

def make_bbox_from_keypoints(
        keypoints=[],
        width = 120,
        height = 120,
        ):
    #Halpe Keypoints for right and left hand selected
    right_hand_keypoints = keypoints[:,-21:,:2]
    left_hand_keypoints = keypoints[:,-42:-21,:2]
    # Create a bounding box for each point on keypoints
    bboxes = []
    for i in range(keypoints.shape[0]):
        right_hand_bboxes = extract_xy_min_max(right_hand_keypoints[i], width, height)
        left_hand_bboxes = extract_xy_min_max(left_hand_keypoints[i], width, height)
        #if no bboxes found for right or left set bbox to image size
        if (right_hand_bboxes<0).any():
            right_hand_bboxes = np.zeros(4)
            right_hand_bboxes[3] = 1500
            right_hand_bboxes[2] = 2040 
        if (left_hand_bboxes<0).any():
            left_hand_bboxes = np.zeros(4)
            left_hand_bboxes[3] = 1500
            left_hand_bboxes[2] = 2040

        bboxes.append([right_hand_bboxes,left_hand_bboxes])

    return 2, bboxes