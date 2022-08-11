import numpy as np


def keypoints_to_bbox(keypoints, thresh=0.1, min_keypoints=5):

    if keypoints.shape[-1] == 3:
        valid = keypoints[:, -1] > thresh
        keypoints = keypoints[valid, :-1]

    if keypoints.shape[0] < min_keypoints:
        return [0.0, 0.0, 0.0, 0.0]

    bbox = [np.min(keypoints[:, 0]), np.min(keypoints[:, 1]), np.max(keypoints[:, 0]), np.max(keypoints[:, 1])]
    bbox = bbox[:2] + [bbox[2] - bbox[0], bbox[3] - bbox[1]]

    return bbox


def compute_iou(box1: np.ndarray, box2: np.ndarray, tlhw=True, epsilon=1e-8):
    """
    calculate intersection over union cover percent

        :param box1: box1 with shape (N,4)
        :param box2: box2 with shape (N,4)
        :tlhw: bool if format is tlhw and need to be converted to tlbr
        :return: IoU ratio if intersect, else 0
    """
    point_num = max(box1.shape[0], box2.shape[0])
    b1p1, b1p2, b2p1, b2p2 = box1[:, :2], box1[:, 2:], box2[:, :2], box2[:, 2:]

    if tlhw:
        b1p2 = b1p1 + b1p2
        b2p2 = b2p1 + b2p2

    # mask that eliminates non-intersecting matrices
    base_mat = np.ones(shape=(point_num,)).astype(float)
    base_mat *= np.all(np.greater(b1p2 - b2p1, 0), axis=1)
    base_mat *= np.all(np.greater(b2p2 - b1p1, 0), axis=1)

    # epsilon handles case where a bbox has zero size (so let's make that have a IoU=0)
    intersect_area = np.prod(np.minimum(b2p2, b1p2) - np.maximum(b1p1, b2p1), axis=1).astype(float)
    union_area = np.prod(b1p2 - b1p1, axis=1) + np.prod(b2p2 - b2p1, axis=1) - intersect_area + epsilon
    intersect_ratio = intersect_area / union_area

    return base_mat * intersect_ratio


def match_keypoints_to_bbox(bbox: np.ndarray, keypoints_list: list, thresh=0.25, num_keypoints=25, visible=True):
    """Finds the best keypoints with an acceptable IoU, if present"""

    if visible:
        empty_keypoints = np.zeros((num_keypoints, 3))
    else:
        empty_keypoints = np.zeros((num_keypoints, 2))

    if keypoints_list is None or len(keypoints_list) == 0:
        return empty_keypoints, None

    bbox = np.reshape(bbox, (1, 4))
    kp_bbox = np.array([keypoints_to_bbox(k) for k in keypoints_list])

    iou = compute_iou(bbox, kp_bbox)
    idx = np.argmax(iou)

    if iou[idx] > thresh:
        return keypoints_list[idx], idx

    return empty_keypoints, None
