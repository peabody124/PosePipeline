import datajoint as dj
from pathlib import Path


def get_pose_project_dir():
    """Dir for local install of pose_pipeline and non-package dependencies. Should return string ending in /"""
    pose_project_dir = dj.config.get("custom", {}).get("pose_project_dir","/home/jcotton/projects/pose/")
    assert Path(pose_project_dir).is_dir(), f"Could not find pose project directory: {pose_project_dir}"
    return pose_project_dir
