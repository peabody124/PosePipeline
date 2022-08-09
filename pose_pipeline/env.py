
import os
import sys
import logging
from .utils.paths import find_full_path
from .paths import get_pose_root_package_dir

logger = logging.getLogger("datajoint")

dependencies = {
    "OpenPose" : {
        "rel_path": "openpose", 
        "env_var" : "OPENPOSE_PATH", 
        "git_repo": "CMU-Perceptual-Computing-Lab/openpose"},
    "OpenPose_build" : {
        "rel_path": "openpose/build/python", 
        "env_var" : "OPENPOSE_PYTHON_PATH", 
        "git_repo": "CMU-Perceptual-Computing-Lab/openpose"},
    "Expose" : {
        "rel_path": "expose", 
        "env_var" : "EXPOSE_PATH", 
        "git_repo": "Jack000/Expose"},
    "CenterHMR" : { # Since renamed ROMP? Issues
        "rel_path": "CenterHMR", 
        "env_var" : "CENTERHMR_PATH", 
        "git_repo": "Arthur151/ROMP"},
    "GAST" : {
        "rel_path": "GAST-Net-3DPoseEstimation", 
        "env_var" : "GAST_PATH", 
        "git_repo": "fabro66/GAST-Net-3DPoseEstimation"},
    "PoseFormer" : {
        "rel_path": "PoseFormer", 
        "env_var" : "POSEFORMER_PATH", 
        "git_repo": "zczcwh/PoseFormer"},
    "VIBE" : {
        "rel_path": "VIBE", 
        "env_var" : "VIBE_PATH", 
        "git_repo": "mkocabas/VIBE"},
    "MEVA" : {
        "rel_path": "MEVA", 
        "env_var" : "MEVA_PATH", 
        "git_repo": "ZhengyiLuo/MEVA"},
    "PARE" : {
        "rel_path": "PARE", 
        "env_var" : "PARE_PATH", 
        "git_repo": "mkocabas/PARE"},
    "PIXIE" : {
        "rel_path": "PIXIE", 
        "env_var" : "PIXIE_PATH", 
        "git_repo": "pixie-io/pixie"},
    "humor" : {
        "rel_path": "humor/humor", 
        "env_var" : "HUMOR_PATH", 
        "git_repo": "davrempe/humor"},
    "FairMOT" : {
        "rel_path": "FairMOT/src/lib", 
        "env_var" : "FAIRMOT_PATH", 
        "git_repo": "ifzhang/FairMOT"},
    "DCNv2" : {
        "rel_path": "DCNv2/DCN", 
        "env_var" : "DCNv2_PATH", 
        "git_repo": "CharlesShang/DCNv2"},
    "TransTrack" : {
        "rel_path": "TransTrack", 
        "env_var" : "TRANSTRACK_PATH", 
        "git_repo": "PeizeSun/TransTrack"},
    "ProHMR" : {
        "rel_path": "ProHMR", 
        "env_var" : "PROHMR_PATH", 
        "git_repo": "nkolot/ProHMR"},
    "TraDeS" : {
        "rel_path": "TraDeS/src/lib", 
        "env_var" : "TRADES_PATH", 
        "git_repo": "yaodongyu/TRADES"},
    "Pose3D-RIE" : {
        "rel_path": "Pose3D-RIE", 
        "env_var" : "RIE_PATH", 
        "git_repo": "paTRICK-swk/Pose3D-RIE"},
    "VideoPose3D" : {
        "rel_path": "VideoPose3D", 
        "env_var" : "VIDEOPOSE3D_PATH", 
        "git_repo": "facebookresearch/VideoPose3D"},
    "PoseAug" : {
        "rel_path": "PoseAug", 
        "env_var" : "POSEAUG_PATH", 
        "git_repo": "jfzhang95/PoseAug"},
}

class add_path():
    """Context function adds path(s) on entry and removes when exiting"""
    def __init__(self, path):
        if not isinstance(path, list):
            self.path = [path]
        else:
            self.path = path

    def __enter__(self):
        for p in self.path:
            sys.path.insert(0, p)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            for p in self.path:
                sys.path.remove(p)
        except ValueError:
            pass

def set_environmental_variables(dependencies=dependencies, git_clone=False):
    """For dependency listed below, checks that path exists and sets env variable.
    
    Relies on `get_pose_root_package_dir` in .paths to determine where packages are.
    If not present, optionally clones the git repository, which relies on git being 
    available as a local OS command.

    Parameters
    ----------
    dependencies: Optional, default from this script. Dict of packages, with values of
                  dicts that specify local rel_path, desired env_var, and git_repo. E.g.
                {"VIBE": {"rel_path": "VIBE", # relative to root from config 
                          "env_var" : "VIBE_PATH", # environment variable
                          "git_repo": "mkocabas/VIBE"}} # github.com implied
    git_clone: Optional, default false. If package is not found, download via git.
    """
    for package in dependencies:
        try:
            package_path = find_full_path(get_pose_root_package_dir(), 
                                          dependencies[package]['rel_path'])
        except FileNotFoundError:
            if git_clone:
                os.system(
                    f"git -C {get_pose_root_package_dir()[0]} clone " + 
                    f"https://github.com/{dependencies[package]['git_repo']}"
                )
                package_path = find_full_path(get_pose_root_package_dir(), 
                                              dependencies[package]['rel_path'])
            else:
                logger.warn(f"Could not find {package}")
                continue
        os.environ[dependencies[package]['env_var']] = package_path

    import platform
    if 'Ubuntu' in platform.version():
        # In Ubuntu, using osmesa mode for rendering
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

