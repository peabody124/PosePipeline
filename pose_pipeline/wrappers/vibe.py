import os
import numpy as np
from pose_pipeline import MODEL_DATA_DIR
from pose_pipeline.utils.bounding_box import get_person_dataloader
from pose_pipeline.utils.bounding_box import convert_crop_coords_to_orig_img, convert_crop_cam_to_orig_img
from pose_pipeline.env import add_path
from pose_pipeline import VideoInfo
import torch


def process_vibe(key):

    crop_size = 224

    spin_checkpoint = os.path.join(MODEL_DATA_DIR, "vibe/spin_model_checkpoint.pth.tar")
    vibe_checkpoint = os.path.join(MODEL_DATA_DIR, "vibe/vibe_model_w_3dpw.pth.tar")

    with add_path(os.environ["VIBE_PATH"]):
        frame_ids, dataloader, bbox = get_person_dataloader(key, crop_size=crop_size)

        from lib.models.vibe import VIBE_Demo

        device = "cuda"
        has_keypoints = False
        model = VIBE_Demo(
            seqlen=16, n_layers=2, hidden_size=1024, add_linear=True, use_residual=True, pretrained=spin_checkpoint
        ).to("cuda")

        ckpt = torch.load(vibe_checkpoint)
        ckpt = ckpt["gen_state_dict"]
        model.load_state_dict(ckpt, strict=False)
        model.eval()

        with torch.no_grad():
            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d, norm_joints2d = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )

            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.unsqueeze(0)
                batch = batch.to(device)

                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]

                pred_cam.append(output["theta"][:, :, :3].reshape(batch_size * seqlen, -1).cpu().detach().numpy())
                pred_verts.append(output["verts"].reshape(batch_size * seqlen, -1, 3).cpu().detach().numpy())
                pred_pose.append(output["theta"][:, :, 3:75].reshape(batch_size * seqlen, -1).cpu().detach().numpy())
                pred_betas.append(output["theta"][:, :, 75:].reshape(batch_size * seqlen, -1).cpu().detach().numpy())
                pred_joints3d.append(output["kp_3d"].reshape(batch_size * seqlen, -1, 3).cpu().detach().numpy())
                smpl_joints2d.append(output["kp_2d"].reshape(batch_size * seqlen, -1, 2).cpu().detach().numpy())

    key["cams"] = np.concatenate(pred_cam, axis=0)
    key["verts"] = np.concatenate(pred_verts, axis=0)
    key["poses"] = np.concatenate(pred_pose, axis=0)
    key["betas"] = np.concatenate(pred_betas, axis=0)
    key["joints3d"] = np.concatenate(pred_joints3d, axis=0)
    key["joints2d"] = np.concatenate(smpl_joints2d, axis=0)

    height, width = (VideoInfo & key).fetch1("height", "width")
    key["cams"] = convert_crop_cam_to_orig_img(key["cams"], bbox, width, height)
    key["joints2d"] = convert_crop_coords_to_orig_img(bbox, key["joints2d"], crop_size)

    return key
