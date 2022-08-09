import datajoint as dj
from collections import abc


def get_pose_root_data_dir():
    pose_data_dirs = dj.config.get("custom", {}).get("pose_root_data_dir")
    if not pose_data_dirs:
        return None
    elif not isinstance(pose_data_dirs, abc.Sequence):
        return list(pose_data_dirs)
    else:
        return pose_data_dirs

def get_pose_root_package_dir():
    pose_package_dirs = dj.config.get("custom", {}).get("pose_root_package_dir")
    if not pose_package_dirs:
        return None
    elif not isinstance(pose_package_dirs, abc.Sequence):
        return list(pose_package_dirs)
    else:
        return pose_package_dirs