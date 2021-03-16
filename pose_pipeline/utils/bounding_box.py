import cv2
import numpy as np


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
