import os
import cv2
import numpy as np
from pose_pipeline import Video, TrackingBbox, PersonBbox


def fix_bb_aspect_ratio(bbox, dilate=1.2, ratio=1.0):
    """Inflates a bounding box with the desired aspect ratio

    Args:
        bbox (4,) : bounding box in TLHW format
        dilate (float): fraction amount to increase for crop
        ratio (float): desired ratio (width / height)

    Returns:
        bbox (4,) : corrected bounding box
    """

    center = bbox[:2] + bbox[2:] / 2.0
    hw = bbox[2:]

    if hw[0] / hw[1] < ratio:
        # if bbox width/height is greater than desired ratio, increase height to match
        hw = np.array([hw[1] * ratio, hw[1]])
    else:
        hw = np.array([hw[0], hw[0] / ratio])
    hw = hw * dilate

    return np.concatenate([center - hw / 2, hw], axis=0)


def crop_image_bbox(image, bbox, target_size=(288, 384), dilate=1.2):
    """Extract the image defined by bounding box with desired aspect ratio

    Args:
        image (np.array): uses HWC format
        bbox (4,): TLHW format bounding box, will contain at least this area
        target_size (optional): image size to produce
        dilate: additional dilation on the bounding box

    Returns:
        cropped image
    """

    bbox = fix_bb_aspect_ratio(bbox, ratio=target_size[0] / target_size[1], dilate=dilate)

    # three points on corner of bounding box
    src = np.asarray([[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]], [bbox[0], bbox[1] + bbox[3]]])
    dst = np.array([[0, 0], [target_size[0], target_size[1]], [0, target_size[1]]])  # .astype(np.float32)
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    image = cv2.warpAffine(image, trans, target_size, flags=cv2.INTER_LINEAR)

    return image, bbox


def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    """
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (TLHW)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:

    Adopted from https://github.com/mkocabas/VIBE/blob/master/lib/utils/demo_utils.py
    """

    cy = bbox[:, 1] + bbox[:, 3] / 2
    cx = bbox[:, 0] + bbox[:, 2] / 2
    h = bbox[:, 2]

    hw, hh = img_width / 2.0, img_height / 2.0
    sx = cam[:, 0] * (1.0 / (img_width / h))
    sy = cam[:, 0] * (1.0 / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:, 1]
    ty = ((cy - hh) / hh / sy) + cam[:, 2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam


def convert_crop_coords_to_orig_img(bbox, keypoints, crop_size):
    # Adopted from https://github.com/mkocabas/VIBE/blob/master/lib/utils/demo_utils.py

    cy = bbox[:, 1] + bbox[:, 3] / 2
    cx = bbox[:, 0] + bbox[:, 2] / 2
    h = bbox[:, 2]

    # unnormalize to crop coords
    keypoints = 0.5 * crop_size * (keypoints + 1.0)

    # rescale to orig img crop
    keypoints *= h[..., None, None] / crop_size

    # transform into original image coords
    keypoints[:, :, 0] = (cx - h / 2)[..., None] + keypoints[:, :, 0]
    keypoints[:, :, 1] = (cy - h / 2)[..., None] + keypoints[:, :, 1]
    return keypoints


def get_person_dataloader(key, batch_size=32, num_workers=16, crop_size=224, scale=1.0):

    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    video, bboxes_dj, present_dj = (Video * PersonBbox & key).fetch1("video", "bbox", "present")

    cap = cv2.VideoCapture(video)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    frames = []
    bboxes = []
    frame_ids = []
    for idx, (bbox, present) in enumerate(zip(bboxes_dj, present_dj)):

        # handle the case where person is not tracked in frame
        if not present:
            ret, frame = cap.read()
            print("Skip missing frame")
            continue

        # should match the length of identified person tracks
        ret, frame = cap.read()
        assert ret and frame is not None

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if type(crop_size) == int or len(crop_size) == 1:
            crop_size = (crop_size, crop_size)

        norm_img, bbox = crop_image_bbox(img, bbox, target_size=crop_size, dilate=scale)
        norm_img = transform(norm_img)

        # print(norm_img.shape)
        # break
        frames.append(norm_img)
        bboxes.append(bbox)
        frame_ids.append(idx)

    cap.release()
    os.remove(video)

    class Inference(Dataset):
        def __init__(self, frames, bboxes=None, joints2d=None):

            self.frames = frames
            self.bboxes = bboxes
            self.joints2d = joints2d
            self.scale = scale
            self.crop_size = crop_size
            self.frames = frames
            self.has_keypoints = True if joints2d is not None else False

            def get_default_transform():

                return transform

            self.transform = get_default_transform()
            self.norm_joints2d = np.zeros_like(self.joints2d)

            if self.has_keypoints:
                bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
                bboxes[:, 2:] = 150.0 / bboxes[:, 2:]
                self.bboxes = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 2]]).T

                self.image_file_names = self.image_file_names[time_pt1:time_pt2]
                self.joints2d = joints2d[time_pt1:time_pt2]
                self.frames = frames[time_pt1:time_pt2]

        def __len__(self):
            return len(self.frames)

        def __getitem__(self, idx):

            img = self.frames[idx]
            j2d = self.joints2d[idx] if self.has_keypoints else None

            if self.has_keypoints:
                return img, kp_2d
            else:
                return img

    dataset = Inference(frames, bboxes)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    return frame_ids, dataloader, np.stack(bboxes, axis=0)
