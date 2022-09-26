import os
import sys
import datajoint as dj
from pathlib import Path
from .utils.paths import find_full_path
from .paths import get_pose_project_dir


class add_path:
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


def set_environmental_variables():
    """For dependency listed below, checks that path exists and sets env variable."""
    POSE_PROJ_DIR = get_pose_project_dir()
    env_paths = {
        "OPENPOSE_PATH": f"{POSE_PROJ_DIR}openpose",
        "OPENPOSE_PYTHON_PATH": f"{POSE_PROJ_DIR}openpose/python",  # removed build/python
        "EXPOSE_PATH": f"{POSE_PROJ_DIR}expose",
        "CENTERHMR_PATH": f"{POSE_PROJ_DIR}CenterHMR",
        "GAST_PATH": f"{POSE_PROJ_DIR}GAST-Net-3DPoseEstimation",
        "POSEFORMER_PATH": f"{POSE_PROJ_DIR}PoseFormer",
        "VIBE_PATH": f"{POSE_PROJ_DIR}VIBE",
        "MEVA_PATH": f"{POSE_PROJ_DIR}MEVA",
        "PARE_PATH": f"{POSE_PROJ_DIR}PARE",
        "PIXIE_PATH": f"{POSE_PROJ_DIR}PIXIE",
        "HUMOR_PATH": f"{POSE_PROJ_DIR}humor/humor",
        "FAIRMOT_PATH": f"{POSE_PROJ_DIR}FairMOT/src/lib",
        "DCNv2_PATH": f"{POSE_PROJ_DIR}DCNv2/DCN",
        "TRANSTRACK_PATH": f"{POSE_PROJ_DIR}TransTrack",
        "PROHMR_PATH": f"{POSE_PROJ_DIR}ProHMR",
        "TRADES_PATH": f"{POSE_PROJ_DIR}TraDeS/src/lib",
        "RIE_PATH": f"{POSE_PROJ_DIR}Pose3D-RIE",
        "VIDEOPOSE3D_PATH": f"{POSE_PROJ_DIR}VideoPose3D",
        "POSEAUG_PATH": f"{POSE_PROJ_DIR}PoseAug",
    }
    for var, path in env_paths.items():
        assert Path(path).exists(), f"Could not find path {path}"
        os.environ[var] = path

    import platform

    if "Ubuntu" in platform.version():
        # In Ubuntu, using osmesa mode for rendering
        os.environ["PYOPENGL_PLATFORM"] = "egl"


def download_git_dependencies():
    """Download git dependency non-packages to pose project dir"""
    req_path = get_pose_project_dir() + "PosePipeline/requirements.txt"
    assert Path(
        req_path
    ).exists(), "Could not find requirements.txt with git repos listed."
    with open(req_path) as f:
        git_repos = f.read().split("\n# git+")[1:]

    for repo in git_repos:
        try:
            find_full_path(get_pose_project_dir(), repo.split("/")[-1])
        except FileNotFoundError:
            os.system(f"git -C {get_pose_project_dir()} clone {repo}")


def pytorch_memory_limit(frac=0.5):
    # limit pytorch memory
    import torch

    torch.cuda.set_per_process_memory_fraction(frac, 0)
    torch.cuda.empty_cache()


def tensorflow_memory_limit():
    # limit tensorflow memory. there are also other approaches
    # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
