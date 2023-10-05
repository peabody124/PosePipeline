import numpy as np
from pose_pipeline.pipeline import VideoInfo


def keypoints_filter_clipped_image(key, keypoints2d):
    """
    Set confidence to zero for any keypoints outsize the image.

    Args:
        key (dict): primary key of the video
        keypoints2d (np.ndarray): keypoints array of shape (N, J, 3) where N is the number of keypoints
    """

    # make sure writeable as DJ defaults to read only
    keypoints2d = keypoints2d.copy()

    height, width = (VideoInfo & key).fetch1('height', 'width')

    # handle any bad projections
    clipped = np.logical_or.reduce(
        (
            keypoints2d[..., 0] <= 0,
            keypoints2d[..., 0] >= width,
            keypoints2d[..., 1] <= 0,
            keypoints2d[..., 1] >= height,
            np.isnan(keypoints2d[..., 0]),
            np.isnan(keypoints2d[..., 1]),
        )
    )
    keypoints2d[clipped, -1] = 0 # modified to work with 3d keypoints

    return keypoints2d

def keypoints_filter_clipped_image3d(key, keypoints2d, keypoints3d):
    """
    Set confidence to zero for any keypoints outsize the image.

    Args:
        key (dict): primary key of the video
        keypoints2d (np.ndarray): keypoints array of shape (N, J, 3) where N is the number of keypoints
    """

    # make sure writeable as DJ defaults to read only
    keypoints2d = keypoints2d.copy()

    height, width = (VideoInfo & key).fetch1('height', 'width')

    # handle any bad projections
    clipped = np.logical_or.reduce(
        (
            keypoints2d[..., 0] <= 0,
            keypoints2d[..., 0] >= width,
            keypoints2d[..., 1] <= 0,
            keypoints2d[..., 1] >= height,
            np.isnan(keypoints2d[..., 0]),
            np.isnan(keypoints2d[..., 1]),
        )
    )
    keypoints3d[clipped, -1] = 0 # modified to work with 3d keypoints

    return keypoints3d