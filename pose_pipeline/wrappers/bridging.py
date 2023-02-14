import os
import cv2
import numpy as np
from pose_pipeline import Video

from pose_pipeline import MODEL_DATA_DIR

import tensorflow as tf
import tensorflow_hub as hub


# supported formats are
# 'smpl_24', 'h36m_17', 'h36m_25', 'mpi_inf_3dhp_17', 'mpi_inf_3dhp_28', 'coco_19', 'sailvos_26', 'gpa_34', 'aspset_17',
# 'bml_movi_87', 'mads_19', 'berkeley_mhad_43', 'total_capture_21', 'jta_22', 'ikea_asm_17', 'human4d_32', 'smplx_42',
# 'ghum_35', 'lsp_14', '3dpeople_29', 'umpm_15', 'kinectv2_25', 'smpl+head_30', ''


def make_coco_25(model):
    # foot keypoints are available in the model, but not listed in the indices
    all_joints = model.per_skeleton_joint_names[""]

    def f(x):
        x = x.decode("utf-8").split("_")[0]
        return x.encode("utf-8")

    coco_idx = [i for i, x in enumerate(all_joints) if "_coco" in x.decode("utf-8")]

    # make sure the new joints are at the end
    new = np.setdiff1d(coco_idx, model.per_skeleton_indices["coco_19"])
    updated = np.concatenate([model.per_skeleton_indices["coco_19"], new])
    model.per_skeleton_indices["coco_25"] = updated

    model.per_skeleton_joint_names["coco_25"] = [f(x) for x in model.per_skeleton_joint_names[""][updated]]
    model.per_skeleton_joint_edges["coco_25"] = model.per_skeleton_joint_edges["coco_19"]

    return model


def get_model():
    # doing this here to only load model once, since this takes quite a while
    if get_model.model is None:
        model_path = os.path.join(MODEL_DATA_DIR, "bridging_formats")
        model = hub.load(model_path)
        # get_model.model = hub.load('https://bit.ly/metrabs_l')  # Takes about 3 minutes

        model.per_skeleton_joint_names = {k: v.numpy() for k, v in model.per_skeleton_joint_names.items()}
        model.per_skeleton_indices = {k: v.numpy() for k, v in model.per_skeleton_indices.items()}

        model = make_coco_25(model)

        get_model.model = model

    return get_model.model


get_model.model = None


def get_joint_names(skeleton, model=None):

    if model is None:
        model = get_model()

    return model.per_skeleton_joint_names[skeleton]


def get_skeleton_edges(skeleton, model=None):

    if model is None:
        model = get_model()

    return model.per_skeleton_joint_edges[skeleton]


def filter_skeleton(keypoints, skeleton, model=None):

    if model is None:
        model = get_model()
    idx = model.per_skeleton_indices[skeleton]

    keypoints = [k[..., idx, :] for k in keypoints]
    return keypoints

def scale_align(poses):
    square_scales = tf.reduce_mean(tf.square(poses), axis=(-2, -1), keepdims=True)
    mean_square_scale = tf.reduce_mean(square_scales, axis=-3, keepdims=True)
    return poses * tf.sqrt(mean_square_scale / square_scales)


def point_stdev(poses, item_axis, coord_axis):
    coordwise_variance = tf.math.reduce_variance(poses, axis=item_axis, keepdims=True)
    average_stdev = tf.sqrt(tf.reduce_sum(coordwise_variance, axis=coord_axis, keepdims=True))
    return tf.squeeze(average_stdev, (item_axis, coord_axis))


def augmentation_noise(poses3d):
    return point_stdev(scale_align(poses3d), item_axis=1, coord_axis=-1).numpy()


def noise_to_conf(x, half_val=200, sharpness=50):
    x = -(x - half_val) / sharpness
    return 1 / (1 + tf.math.exp(-x))


def bridging_formats_bottom_up(key, model=None, skeleton=""):

    if model is None:
        model = get_model()

    from mmpose.apis import init_pose_model, inference_bottom_up_pose_model
    from tqdm import tqdm

    video = Video.get_robust_reader(key, return_cap=False)
    cap = cv2.VideoCapture(video)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    boxes = []
    keypoints2d = []
    keypoints3d = []
    keypoint_noises = []
    for _ in tqdm(range(video_length)):

        # should match the length of identified person tracks
        ret, frame = cap.read()
        assert ret and frame is not None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pred = model.detect_poses(frame, skeleton=skeleton, num_aug=10, average_aug=False)

        boxes.append(pred["boxes"].numpy())
        keypoints2d.append(np.mean(pred["poses2d"].numpy(), axis=1))
        keypoints3d.append(np.mean(pred["poses3d"].numpy(), axis=1))
        keypoint_noises.append(augmentation_noise(pred["poses3d"].numpy()))

    cap.release()
    os.remove(video)

    return {"boxes": boxes, "keypoints2d": keypoints2d, "keypoints3d": keypoints3d, "keypoint_noise": keypoint_noises}


def get_overlay_callback(boxes, poses2d, joint_edges=None):
    def overlay_callback(image, idx):
        image = image.copy()
        bbox = boxes[idx]  # boxes is frames x 5
        p2d = poses2d[idx]  # poses2d is frames x 2
        small = int(5e-3 * np.max(image.shape))

        for bbox, p2d in zip(bbox, p2d):

            cv2.rectangle(
                image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])),
                (255, 255, 255),
                small,
            )

            if joint_edges is not None:
                for i_start, i_end in joint_edges:
                    cv2.line(
                        image,
                        (int(p2d[i_start, 0]), int(p2d[i_start, 1])),
                        (int(p2d[i_end, 0]), int(p2d[i_end, 1])),
                        (0, 200, 100),
                        thickness=4,
                    )

            for x, y in p2d:
                cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), thickness=3)

        return image

    return overlay_callback


normalized_joint_name_dictionary = {
    "coco_25": [
        "Neck",
        "Nose",
        "Pelvis",
        "Left Shoulder",
        "Left Elbow",
        "Left Wrist",
        "Left Hip",
        "Left Knee",
        "Left Ankle",
        "Right Shoulder",
        "Right Elbow",
        "Right Wrist",
        "Right Hip",
        "Right Knee",
        "Right Ankle",
        "Left Eye",
        "Left Ear",
        "Right Eye",
        "Right Ear",
        "Left Big Toe",  # caled lfoo in the code
        "Left Small Toe",
        "Left Heel",
        "Right Big Toe",
        "Right Small Toe",
        "Right Heel",
    ],
    "bml_movi_87": [
        "backneck",
        "upperback",
        "clavicle",
        "sternum",
        "umbilicus",
        "lfronthead",
        "lbackhead",
        "lback",
        "lshom",
        "lupperarm",
        "lelbm",
        "lforearm",
        "lwrithumbside",
        "lwripinkieside",
        "lfin",
        "lasis",
        "lpsis",
        "lfrontthigh",
        "lthigh",
        "lknem",
        "lankm",
        "Left Heel",  # "lhee",
        "lfifthmetatarsal",
        "Left Big Toe",  # "ltoe",
        "lcheek",
        "lbreast",
        "lelbinner",
        "lwaist",
        "lthumb",
        "lfrontinnerthigh",
        "linnerknee",
        "lshin",
        "lfirstmetatarsal",
        "lfourthtoe",
        "lscapula",
        "lbum",
        "rfronthead",
        "rbackhead",
        "rback",
        "rshom",
        "rupperarm",
        "relbm",
        "rforearm",
        "rwrithumbside",
        "rwripinkieside",
        "rfin",
        "rasis",
        "rpsis",
        "rfrontthigh",
        "rthigh",
        "rknem",
        "rankm",
        "Right Heel",  # "rhee",
        "rfifthmetatarsal",
        "Right Big Toe",  # "rtoe",
        "rcheek",
        "rbreast",
        "relbinner",
        "rwaist",
        "rthumb",
        "rfrontinnerthigh",
        "rinnerknee",
        "rshin",
        "rfirstmetatarsal",
        "rfourthtoe",
        "rscapula",
        "rbum",
        "Head",  # "head",
        "mhip",
        "Pelvis",  # "pelv",
        "Sternum",  # "thor",
        "Left Ankle",  # "lank",
        "Left Elbow",  # "lelb",
        "Left Hip",  # "lhip",
        "Left Hand",  # "lhan",
        "Left Knee",  # "lkne",
        "Left Shoulder",  # "lsho",
        "Left Wrist",  # "lwri",
        "Left Foot",  # "lfoo",
        "Right Ankle",  # "rank",
        "Right Elbow",  # "relb",
        "Right Hip",  # "rhip",
        "Right Hand",  # "rhan",
        "Right Knee",  # "rkne",
        "Right Shoulder",  # "rsho",
        "Right Wrist",  # "rwri",
        "Right Foot",  # "rfoo",
    ],
}
