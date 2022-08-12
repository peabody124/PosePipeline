def lifting_kinematics(key):
    import numpy as np

    keypoints3d = (LiftingPerson & key).fetch1("keypoints_3d")
    keypoints = (TopDownPerson & key).fetch1("keypoints")
    timestamps = (VideoInfo & key).fetch1("timestamps")

    leg_idx = np.array(
        [
            TopDownPerson.joint_names().index(k)
            for k in ["Left Ankle", "Left Knee", "Left Hip", "Right Hip", "Right Knee", "Right Ankle"]
        ]
    )
    keypoints_valid = np.all(keypoints[:, leg_idx, -1] > 0.5, axis=1)
    #        keypoints_valid = np.arange(np.where(keypoints_valid)[0][0], np.where(keypoints_valid)[0][-1]+1)
    keypoints3d = keypoints3d[keypoints_valid]

    timestamps = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])[np.where(keypoints_valid)[0]]

    idx = [GastNetPerson.joint_names().index(j) for j in ["Right hip", "Left hip"]]

    delta_pelvis = keypoints3d[:, idx[1]] - keypoints3d[:, idx[0]]
    pelvis_angle = -np.arctan2(delta_pelvis[:, 0], delta_pelvis[:, 1])
    pelvis_angle = np.unwrap(pelvis_angle)

    pelvis_angle = np.median(pelvis_angle, axis=0, keepdims=True)

    z = np.zeros(pelvis_angle.shape)
    pelvis_rot = np.array(
        [
            [np.cos(pelvis_angle), -np.sin(pelvis_angle), z],
            [np.sin(pelvis_angle), np.cos(pelvis_angle), z],
            [z, z, 1 + z],
        ]
    )
    pelvis_rot = np.transpose(pelvis_rot, [2, 0, 1])

    # derotate the points
    keypoints3d = keypoints3d @ pelvis_rot

    # start collation outputs
    joint_names = LiftingPerson.joint_names()
    outputs = {
        "timestamps": timestamps,
        "Right Foot": keypoints3d[:, joint_names.index("Right foot"), 0],
        "Left Foot": keypoints3d[:, joint_names.index("Left foot"), 0],
    }

    # pick the joints to extract from GastNet in the sagital plane
    angles = [
        ("Right Hip", ("Right hip", "Right knee"), ("Spine", "Hip (root)")),
        ("Left Hip", ("Left hip", "Left knee"), ("Spine", "Hip (root)")),
        ("Right Knee", ("Right knee", "Right foot"), ("Right hip", "Right knee")),
        ("Left Knee", ("Left knee", "Left foot"), ("Left hip", "Left knee")),
    ]
    plane = np.array([0, 2])

    for joint in angles:
        joint, v1, v2 = joint

        # compute the difference between two joint locations in the sagital plane
        v1 = keypoints3d[:, joint_names.index(v1[1]), plane] - keypoints3d[:, joint_names.index(v1[0]), plane]
        v2 = keypoints3d[:, joint_names.index(v2[1]), plane] - keypoints3d[:, joint_names.index(v2[0]), plane]

        v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
        v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)
        angle = np.arccos(np.sum(v1 * v2, axis=-1)) * 180 / np.pi

        outputs[joint] = angle

    return outputs
