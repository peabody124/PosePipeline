import datajoint as dj
from pathlib import Path
from collections import abc


def get_pose_root_data_dir():
    pose_data_dirs = dj.config.get("custom", {}).get("pose_root_data_dir")
    if not pose_data_dirs:
        return None
    elif not isinstance(pose_data_dirs, abc.Sequence):
        return list(pose_data_dirs)
    else:
        return pose_data_dirs


def get_pose_project_dir():
    """Dir for local install of pose_pipeline and non-package dependencies. Should return string ending in /"""
    pose_project_dir = dj.config.get("custom", {}).get("pose_project_dir")
    assert Path(pose_project_dir).is_dir(), f"Could not find pose project directory: {pose_project_dir}"
    return pose_project_dir
