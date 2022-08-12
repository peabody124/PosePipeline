import os
import numpy as np
from tqdm import tqdm
from pose_pipeline import MODEL_DATA_DIR
from pose_pipeline.utils.bounding_box import get_person_dataloader
from pose_pipeline.utils.bounding_box import (
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
)
from pose_pipeline.env import add_path
from pose_pipeline import VideoInfo, OpenPosePerson, TopDownPerson, TopDownMethodLookup
import torch


def batch_rot2aa(Rs, epsilon=1e-7):
    """
    Rs is B x 3 x 3
    void cMathUtil::RotMatToAxisAngle(const tMatrix& mat, tVector& out_axis,
                                      double& out_theta)
    {
        double c = 0.5 * (mat(0, 0) + mat(1, 1) + mat(2, 2) - 1);
        c = cMathUtil::Clamp(c, -1.0, 1.0);
        out_theta = std::acos(c);
        if (std::abs(out_theta) < 0.00001)
        {
            out_axis = tVector(0, 0, 1, 0);
        }
        else
        {
            double m21 = mat(2, 1) - mat(1, 2);
            double m02 = mat(0, 2) - mat(2, 0);
            double m10 = mat(1, 0) - mat(0, 1);
            double denom = std::sqrt(m21 * m21 + m02 * m02 + m10 * m10);
            out_axis[0] = m21 / denom;
            out_axis[1] = m02 / denom;
            out_axis[2] = m10 / denom;
            out_axis[3] = 0;
        }
    }
    """

    cos = 0.5 * (torch.einsum("bii->b", [Rs]) - 1)
    cos = torch.clamp(cos, -1 + epsilon, 1 - epsilon)

    theta = torch.acos(cos)

    m21 = Rs[:, 2, 1] - Rs[:, 1, 2]
    m02 = Rs[:, 0, 2] - Rs[:, 2, 0]
    m10 = Rs[:, 1, 0] - Rs[:, 0, 1]
    denom = torch.sqrt(m21 * m21 + m02 * m02 + m10 * m10 + epsilon)

    axis0 = torch.where(torch.abs(theta) < 0.00001, m21, m21 / denom)
    axis1 = torch.where(torch.abs(theta) < 0.00001, m02, m02 / denom)
    axis2 = torch.where(torch.abs(theta) < 0.00001, m10, m10 / denom)

    return theta.unsqueeze(1) * torch.stack([axis0, axis1, axis2], 1)


def process_prohmr(key, optimization=True, batch_size=1):

    model_path = os.path.join(MODEL_DATA_DIR, "prohmr/")
    joint_regressor_extra = os.path.join(MODEL_DATA_DIR, "prohmr/SMPL_to_J19.pkl")
    mean_params = os.path.join(MODEL_DATA_DIR, "prohmr/smpl_mean_params.npz")
    checkpoint = os.path.join(MODEL_DATA_DIR, "prohmr/checkpoint.pt")

    results = []

    height, width = (VideoInfo & key).fetch1("height", "width")

    with add_path(os.environ["PROHMR_PATH"]):

        from prohmr.configs import prohmr_config
        from prohmr.models import ProHMR
        from prohmr.optimization import KeypointFitting
        from prohmr.utils import recursive_to
        from prohmr.utils.geometry import perspective_projection

        model_cfg = prohmr_config()
        model_cfg["SMPL"]["MODEL_PATH"] = model_path
        model_cfg["SMPL"]["JOINT_REGRESSOR_EXTRA"] = joint_regressor_extra
        model_cfg["SMPL"]["MEAN_PARAMS"] = mean_params

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        model = ProHMR.load_from_checkpoint(checkpoint, strict=False, cfg=model_cfg).to(device)
        model.eval()

        if optimization:
            keypoint_fitting = KeypointFitting(model_cfg)
            keypoints = (OpenPosePerson & key).fetch1("keypoints")

        crop_size = model_cfg["MODEL"]["IMAGE_SIZE"]
        frame_ids, dataloader, bbox = get_person_dataloader(key, crop_size=crop_size, batch_size=batch_size)

        frames = []
        pred_cam = []
        pred_verts = []
        pred_pose = []
        pred_betas = []
        pred_joints3d = []
        smpl_joints2d = []

        for idx, batch in tqdm(enumerate(dataloader)):

            frame_id = frame_ids[idx]

            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model({"img": batch})

            if optimization:

                kp = np.concatenate([keypoints[None, frame_id, ...], np.zeros((1, 19, 3))], axis=1)

                scale = np.array([width, height]) / 2.0
                center = np.array([width, height]) / 2.0

                batch_opt = {
                    "img": batch.to(device),
                    "orig_keypoints_2d": torch.from_numpy(kp).to(device),
                    "box_center": torch.from_numpy(center[None, ...]).to(device),
                    "box_size": torch.from_numpy(scale[None, 0]).to(device),
                    "img_size": torch.from_numpy(np.array([[width, height]])).to(device),
                }

                opt_out = model.downstream_optimization(
                    regression_output=out,
                    batch=batch_opt,
                    opt_task=keypoint_fitting,
                    use_hips=False,
                    full_frame=True,
                )

                camera_center = torch.tensor([[width / 2, height / 2]])
                projected_joints = perspective_projection(
                    opt_out["model_joints"],
                    opt_out["camera_translation"],
                    model_cfg["EXTRA"]["FOCAL_LENGTH"] * torch.ones_like(camera_center),
                    camera_center=camera_center,
                )
                projected_joints = projected_joints.detach().cpu().numpy()

                frames.append(frame_id)

                pred_cam.append(opt_out["camera_translation"].cpu().detach().numpy())
                pred_verts.append(opt_out["vertices"].cpu().detach().numpy())

                # for consistency with other algos and rendering code
                global_orient = opt_out["smpl_params"]["global_orient"][0]
                global_orient = batch_rot2aa(global_orient)
                poses_aa = batch_rot2aa(opt_out["smpl_params"]["body_pose"][0]).reshape(1, 23 * 3)
                pred_pose.append(torch.cat([global_orient, poses_aa], axis=-1).cpu().detach().numpy())

                pred_betas.append(opt_out["smpl_params"]["betas"].cpu().detach().numpy())
                pred_joints3d.append(opt_out["model_joints"].cpu().detach().numpy())

                smpl_joints2d.append(projected_joints)

            else:
                print("Not implemented")

    key["cams"] = np.concatenate(pred_cam, axis=0)
    key["verts"] = np.concatenate(pred_verts, axis=0)
    key["poses"] = np.concatenate(pred_pose, axis=0)
    key["betas"] = np.concatenate(pred_betas, axis=0)
    key["joints3d"] = np.concatenate(pred_joints3d, axis=0)
    key["joints2d"] = np.concatenate(smpl_joints2d, axis=0)

    N = key["cams"].shape[0]

    return key


def process_prohmr_mmpose(key, batch_size=1):

    top_down_method = (TopDownMethodLookup & 'top_down_method_name="MMPose"').fetch1("top_down_method")
    assert len(TopDownPerson & key & f"top_down_method={top_down_method}") == 1

    model_path = os.path.join(MODEL_DATA_DIR, "prohmr/")
    mean_params = os.path.join(MODEL_DATA_DIR, "prohmr/smpl_mean_params.npz")
    checkpoint = os.path.join(MODEL_DATA_DIR, "prohmr/checkpoint.pt")
    smpl_to_mmpose = [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]

    results = []

    height, width = (VideoInfo & key).fetch1("height", "width")

    with add_path(os.environ["PROHMR_PATH"]):

        from prohmr.configs import prohmr_config
        from prohmr.models import ProHMR
        from prohmr.optimization import KeypointFitting
        from prohmr.utils import recursive_to
        from prohmr.utils.geometry import perspective_projection

        model_cfg = prohmr_config()
        model_cfg["SMPL"]["MODEL_PATH"] = model_path
        model_cfg["SMPL"]["JOINT_REGRESSOR_EXTRA"] = None
        model_cfg["SMPL"]["JOINT_MAP"] = smpl_to_mmpose
        model_cfg["SMPL"]["MEAN_PARAMS"] = mean_params

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        model = ProHMR.load_from_checkpoint(checkpoint, strict=False, cfg=model_cfg).to(device)
        model.eval()

        keypoint_fitting = KeypointFitting(model_cfg)
        keypoints = (TopDownPerson & key & f"top_down_method={top_down_method}").fetch1("keypoints")

        crop_size = model_cfg["MODEL"]["IMAGE_SIZE"]
        frame_ids, dataloader, bbox = get_person_dataloader(key, crop_size=crop_size, batch_size=batch_size)

        frames = []
        pred_cam = []
        pred_verts = []
        pred_pose = []
        pred_betas = []
        pred_joints3d = []
        smpl_joints2d = []

        for idx, batch in tqdm(enumerate(dataloader)):

            frame_id = frame_ids[idx]

            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model({"img": batch})

            kp = keypoints[None, frame_id, ...]

            scale = np.array([width, height]) / 2.0
            center = np.array([width, height]) / 2.0

            batch_opt = {
                "img": batch.to(device),
                "orig_keypoints_2d": torch.from_numpy(kp).to(device),
                "box_center": torch.from_numpy(center[None, ...]).to(device),
                "box_size": torch.from_numpy(scale[None, 0]).to(device),
                "img_size": torch.from_numpy(np.array([[width, height]])).to(device),
            }

            opt_out = model.downstream_optimization(
                regression_output=out,
                batch=batch_opt,
                opt_task=keypoint_fitting,
                use_hips=True,
                full_frame=True,
            )

            camera_center = torch.tensor([[width / 2, height / 2]])
            projected_joints = perspective_projection(
                opt_out["model_joints"],
                opt_out["camera_translation"],
                model_cfg["EXTRA"]["FOCAL_LENGTH"] * torch.ones_like(camera_center),
                camera_center=camera_center,
            )
            projected_joints = projected_joints.detach().cpu().numpy()

            frames.append(frame_id)

            pred_cam.append(opt_out["camera_translation"].cpu().detach().numpy())
            pred_verts.append(opt_out["vertices"].cpu().detach().numpy())

            # for consistency with other algos and rendering code
            global_orient = opt_out["smpl_params"]["global_orient"][0]
            global_orient = batch_rot2aa(global_orient)
            poses_aa = batch_rot2aa(opt_out["smpl_params"]["body_pose"][0]).reshape(1, 23 * 3)
            pred_pose.append(torch.cat([global_orient, poses_aa], axis=-1).cpu().detach().numpy())

            pred_betas.append(opt_out["smpl_params"]["betas"].cpu().detach().numpy())
            pred_joints3d.append(opt_out["model_joints"].cpu().detach().numpy())

            smpl_joints2d.append(projected_joints)

    key["cams"] = np.concatenate(pred_cam, axis=0)
    key["verts"] = np.concatenate(pred_verts, axis=0)
    key["poses"] = np.concatenate(pred_pose, axis=0)
    key["betas"] = np.concatenate(pred_betas, axis=0)
    key["joints3d"] = np.concatenate(pred_joints3d, axis=0)
    key["joints2d"] = np.concatenate(smpl_joints2d, axis=0)

    N = key["cams"].shape[0]

    return key


def get_prohmr_smpl_callback(key, poses, betas, cams):

    from pose_estimation.body_models.smpl import SMPL
    from pose_pipeline import VideoInfo, PersonBbox, SMPLPerson
    from pose_pipeline.utils.visualization import draw_keypoints

    with add_path(os.environ["PROHMR_PATH"]):
        from prohmr.utils.renderer import Renderer
        from prohmr.configs import prohmr_config

        import torchvision.transforms as transforms

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

        height, width = (VideoInfo & key).fetch1("height", "width")

        valid_idx = np.where((PersonBbox & key).fetch1("present"))[0]

        smpl = SMPL()
        verts = smpl(poses, betas)[0].numpy()

        renderer = Renderer(prohmr_config(), faces=smpl.get_faces())

        joints2d = (SMPLPerson & key).fetch1("joints2d")

        def overlay(frame, idx, renderer=renderer, verts=verts, cams=cams, joints2d=joints2d):

            frame = transform(frame)

            smpl_idx = np.where(valid_idx == idx)[0]
            if len(smpl_idx) == 1:
                smpl_idx = smpl_idx[0]

                frame = renderer(verts[smpl_idx], cams[smpl_idx].copy(), frame)
                frame = (frame * 255).astype(np.uint8)
                frame = draw_keypoints(frame, joints2d[smpl_idx], radius=4)
            else:
                frame = np.transpose(frame.detach().cpu().numpy() * 0.5 + 0.5, [1, 2, 0])
                frame = (frame * 255).astype(np.uint8)

            return frame

        return overlay
