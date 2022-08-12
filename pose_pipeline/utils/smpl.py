import numpy as np
from scipy.spatial.transform import Rotation as R


# from https://github.com/nkolot/SPIN/blob/master/constants.py#L14
JOINT_NAMES_49 = [
    # 25 OpenPose joints (in the order provided by OpenPose)
    "OP Nose",
    "OP Neck",
    "OP RShoulder",
    "OP RElbow",
    "OP RWrist",
    "OP LShoulder",
    "OP LElbow",
    "OP LWrist",
    "OP MidHip",
    "OP RHip",
    "OP RKnee",
    "OP RAnkle",
    "OP LHip",
    "OP LKnee",
    "OP LAnkle",
    "OP REye",
    "OP LEye",
    "OP REar",
    "OP LEar",
    "OP LBigToe",
    "OP LSmallToe",
    "OP LHeel",
    "OP RBigToe",
    "OP RSmallToe",
    "OP RHeel",
    # 24 Ground Truth joints (superset of joints from different datasets)
    "Right Ankle",
    "Right Knee",
    "Right Hip",
    "Left Hip",
    "Left Knee",
    "Left Ankle",
    "Right Wrist",
    "Right Elbow",
    "Right Shoulder",
    "Left Shoulder",
    "Left Elbow",
    "Left Wrist",
    "Neck (LSP)",
    "Top of Head (LSP)",
    "Pelvis (MPII)",
    "Thorax (MPII)",
    "Spine (H36M)",
    "Jaw (H36M)",
    "Head (H36M)",
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Ear",
    "Right Ear",
]


def rotation_6d_to_matrix(d6):
    # adopted from pytorch implementation
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1)[..., None] * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def to_rotvec(x):
    batch, joints, _, _ = x.shape
    x = x.reshape([batch * joints, 3, 3])
    x = R.from_matrix(x).as_rotvec()
    x = x.reshape([batch, joints, 3])
    return x


def convert_pixie_pose_to_smpl(pose):

    import torch
    import os
    from pose_pipeline import add_path

    partbody_pose = pose["partbody_pose"].reshape([-1, 17, 6])
    full_pose = np.concatenate(
        [
            partbody_pose[:, :11],
            pose["neck_pose"][:, None],
            partbody_pose[:, 11 : 11 + 2],
            pose["head_pose"][:, None],
            partbody_pose[:, 13 : 13 + 4],
            pose["left_wrist_pose"][:, None],
            pose["right_wrist_pose"][:, None],
        ],
        axis=1,
    )

    with add_path(os.environ["PIXIE_PATH"]):
        from pixielib.utils import rotation_converter as converter

    full_pose = converter.batch_cont2matrix(torch.Tensor(full_pose)).cpu().detach().numpy()

    pose_rotvec = to_rotvec(full_pose)
    return pose_rotvec
