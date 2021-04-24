import os
import cv2
import numpy as np
from pose_pipeline import Video, TrackingBbox, PersonBboxValid


def fix_bb_aspect_ratio(bbox, dilate=1.2, ratio=1.0):
    """ Inflates a bounding box with the desired aspect ratio 
    
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
    
    return np.concatenate([center - hw/2, hw], axis=0)


def crop_image_bbox(image, bbox, target_size=(288, 384), dilate=1.2):
    """ Extract the image defined by bounding box with desired aspect ratio
    
        Args:
            image (np.array): uses HWC format
            bbox (4,): bounding box, will contain at least this area
            target_size (optional): image size to produce
            dilate: additional dilation on the bounding box
            
        Returns:
            cropped image
    """

    bbox = fix_bb_aspect_ratio(bbox, ratio=target_size[0]/target_size[1], dilate=dilate)
    
    # three points on corner of bounding box
    src = np.asarray([[bbox[0], bbox[1]], [bbox[0]+bbox[2], bbox[1]+bbox[3]], [bbox[0], bbox[1]+bbox[3]]])
    dst = np.array([[0, 0], [target_size[0], target_size[1]], [0, target_size[1]]])
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    image = cv2.warpAffine(image, trans, target_size, flags=cv2.INTER_LINEAR)

    return image, bbox


def get_person_dataloader(key, batch_size=32, num_workers=16, crop_size=224):

    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    video, tracks, keep_tracks = (Video * TrackingBbox * PersonBboxValid & key).fetch1('video', 'tracks', 'keep_tracks')

    cap = cv2.VideoCapture(video)

    frames = []
    bboxes = []
    frame_ids = []
    for i, idx in enumerate(range(len(tracks))):
        bbox = [t['tlhw'] for t in tracks[idx] if t['track_id'] in keep_tracks]

        # handle the case where person is not tracked in frame
        if len(bbox) == 0:
            continue

        # should match the length of identified person tracks
        ret, frame = cap.read()
        assert ret and frame is not None

        frames.append(frame)
        bboxes.append(bbox)
        frame_ids.append(idx)


    class Inference(Dataset):
        def __init__(self, frames, bboxes=None, joints2d=None, scale=1.0, crop_size=crop_size):

            self.frames = frames
            self.bboxes = bboxes
            self.joints2d = joints2d
            self.scale = scale
            self.crop_size = crop_size
            self.frames = frames
            self.has_keypoints = True if joints2d is not None else False

            def get_default_transform():
                normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
                return transform

            self.transform = get_default_transform()
            self.norm_joints2d = np.zeros_like(self.joints2d)

            if self.has_keypoints:
                bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
                bboxes[:, 2:] = 150. / bboxes[:, 2:]
                self.bboxes = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 2]]).T

                self.image_file_names = self.image_file_names[time_pt1:time_pt2]
                self.joints2d = joints2d[time_pt1:time_pt2]
                self.frames = frames[time_pt1:time_pt2]

        def __len__(self):
            return len(self.frames)

        def __getitem__(self, idx):

            img = cv2.cvtColor(self.frames[idx], cv2.COLOR_BGR2RGB)
            bbox = self.bboxes[idx][0]
            j2d = self.joints2d[idx] if self.has_keypoints else None

            norm_img = crop_image_bbox(img, bbox, target_size=(self.crop_size, self.crop_size), dilate=self.scale)[0]
            norm_img = self.transform(norm_img)

            if self.has_keypoints:
                return norm_img, kp_2d
            else:
                return norm_img

    dataset = Inference(frames, bboxes)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    
    return frame_ids, dataloader

