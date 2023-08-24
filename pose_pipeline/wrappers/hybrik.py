import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from pose_pipeline import MODEL_DATA_DIR, VideoInfo, SMPLPerson
from pose_pipeline.utils.bounding_box import get_person_dataloader
from pose_pipeline.env import add_path

config = '/home/jcotton/projects/pose/HybrIK/configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml'
pretrained_weights = '/home/jcotton/projects/pose/HybrIK/model_files/pretrained_hrnet.pth'


def concatenate_dict(accumulator, new_data):

    if len(accumulator) == 0:
        return new_data.copy()

    for k in new_data.keys():
        if type(new_data[k]) == dict:
            accumulator[k] = concatenate_dict(accumulator[k], new_data[k])
        else:
            accumulator[k] = np.concatenate([accumulator[k], new_data[k]], axis=0)

    return accumulator


def finalize_dict(accumulator, accumulated_frames, num_frames):

    frame_ids = np.arange(num_frames)
    found_ids = np.isin(frame_ids, accumulated_frames)

    clean_dict = {}
    for k in accumulator.keys():
        if type(accumulator[k]) == dict:
            clean_dict[k] = finalize_dict(accumulator[k], accumulated_frames, num_frames)
        else:
            clean_dict[k] = np.zeros((num_frames, *accumulator[k].shape[1:])) * np.nan
            clean_dict[k][found_ids] = accumulator[k]

    return clean_dict


def load_hybrik():

    # hacky but required for now
    wd = os.getcwd()

    os.chdir(os.environ["HYBRIDIK_PATH"])

    with add_path(os.environ["HYBRIDIK_PATH"]):
        from hybrik.models import builder
        from hybrik.utils.config import update_config

        cfg = update_config(config)
        hybrik_model = builder.build_sppe(cfg.MODEL)

    save_dict = torch.load(pretrained_weights, map_location='cpu')
    if type(save_dict) == dict:
        model_dict = save_dict['model']
        hybrik_model.load_state_dict(model_dict)
    else:
        hybrik_model.load_state_dict(save_dict)

    os.chdir(wd)

    return hybrik_model


def process_batch(images, bboxes_tlhw, centers, hybrik_model, device='cuda'):
    from hybrik.models.layers.smpl.lbs import rotmat_to_quat, hybrik as hybrik_compute_smpl

    bboxes_tlbr = np.concatenate([bboxes_tlhw[:, :2],
                                  bboxes_tlhw[:, :2] + bboxes_tlhw[:, 2:]], axis=1)

    with torch.no_grad():

        res = hybrik_model(images.to(device),
                        bboxes=torch.Tensor(bboxes_tlbr).to(device),
                        img_center=torch.Tensor(centers).to(device))

        smpl = hybrik_model.smpl

        betas = res.pred_shape
        global_orient = None
        pose_skeleton = res.pred_xyz_jts_29.type(smpl.dtype) * hybrik_model.depth_factor
        pose_skeleton = pose_skeleton.reshape((pose_skeleton.shape[0], -1, 3))
        phis = res.pred_phi.type(smpl.dtype)
        leaf_thetas = None

        vertices, new_joints, rot_mats, joints_from_verts = hybrik_compute_smpl(
            betas, global_orient, pose_skeleton, phis,
            smpl.v_template, smpl.shapedirs, smpl.posedirs,
            smpl.J_regressor, smpl.J_regressor_h36m, smpl.parents, smpl.children_map,
            smpl.lbs_weights, dtype=smpl.dtype, train=False,
            leaf_thetas=leaf_thetas)

        widths = bboxes_tlhw[:, 2]
        focal = 1000.0 / 256.0 * widths

        joints2d = res.pred_uvd_jts.reshape([res.pred_uvd_jts.shape[0], -1, 3]).detach().cpu().numpy()
        joints2d = joints2d * bboxes_tlhw[:, None, None, 2]
        joints2d[:, :, 0] = joints2d[:, :, 0] + bboxes_tlhw[:, None, 0] + bboxes_tlhw[:, None, 2] / 2
        joints2d[:, :, 1] = joints2d[:, :, 1] + bboxes_tlhw[:, None, 1] + bboxes_tlhw[:, None, 2] / 2

        wrapper_entry = {
            'cams': {
                'transl': res.transl.detach().cpu().numpy(),
                'focal': focal,

                # Note: workaround. required for recomputing with current code.
                'phis': res.pred_phi.detach().cpu().numpy(),

                # TODO: this really shouldn't be stored here. takes up tons of unecessary space
                #'vertices': vertices.detach().cpu().numpy(),
            },
            'poses': rot_mats.detach().cpu().numpy(),
            'betas': res.pred_shape.detach().cpu().numpy(),
            'joints2d': joints2d,
            'joints3d': res.pred_xyz_jts_29.reshape([res.pred_xyz_jts_29.shape[0], -1, 3]).detach().cpu().numpy(),
        }

        res.joints2d = res.pred_uvd_jts.reshape([res.pred_uvd_jts.shape[0], -1, 3])
        res.joints3d = res.pred_xyz_jts_29.reshape([res.pred_xyz_jts_29.shape[0], -1, 3])

    return wrapper_entry, res


def process_hybrik(key):

    frame_ids, dataloader, bboxes = get_person_dataloader(key, crop_size=(256, 256))

    hybrik_model = load_hybrik()
    hybrik_model.to('cuda');

    num_frames, height, width = (VideoInfo & key).fetch1('num_frames', 'height', 'width')

    idx = 0
    entry, all_res = {}, {}

    for images in tqdm(dataloader):
        nframes = images.shape[0]
        wrapper_entry, res = process_batch(images, bboxes[idx:idx+nframes],
                                            centers=np.tile(np.array([width/2, height/2]), [nframes, 1]),
                                            hybrik_model=hybrik_model)
        idx = idx + nframes

        entry = concatenate_dict(entry, wrapper_entry)

        for k in res.keys():
            res[k] = res[k].detach().cpu().numpy()
        all_res = concatenate_dict(all_res, res)

    # ensure any frames we skip exist in the final data
    entry = finalize_dict(entry, frame_ids, num_frames)

    entry.update(key)

    return entry


def get_hybrik_smpl_callback(key, device='cuda'):

    with add_path(os.environ["HYBRIDIK_PATH"]):
        from hybrik.models.layers.smpl.lbs import hybrik as hybrik_compute_smpl
        from hybrik.utils.render_pytorch3d import render_mesh

    from pose_pipeline.utils.visualization import draw_keypoints

    hybrik_model = load_hybrik()
    hybrik_model.to(device)
    smpl = hybrik_model.smpl

    poses, betas, cams, joints2d, joints3d = (SMPLPerson & key).fetch1('poses', 'betas', 'cams', 'joints2d', 'joints3d')

    joints2d = joints2d.copy()
    joints2d[:, :, 2] = 1.0

    transl = cams['transl']
    focal = cams['focal']
    phis = cams['phis']

    smpl_faces = torch.from_numpy(hybrik_model.smpl.faces.astype(np.int32))

    def f(x):
        return torch.Tensor(x.copy()).to(device)

    def overlay(image, idx):

        if np.any(np.isnan(joints3d[idx])):
            return image

        with torch.no_grad():
            output = hybrik_model.smpl.hybrik(
                pose_skeleton=f(joints3d[idx:idx+1]), # unit: meter
                betas=f(betas[idx:idx+1]),
                phis=f(phis[idx:idx+1]),
                global_orient=None,
                return_verts=True
            )
            vertices = output.vertices

            # this code is all written like it works on a batch but actually just works
            # on the desired frame
            color_batch = render_mesh(vertices=vertices[:1],
                                      faces=smpl_faces,
                                      translation=f(transl[idx:idx+1]),
                                      focal_length=focal[idx], height=image.shape[0], width=image.shape[1])

            valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
            image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
            image_vis_batch = (image_vis_batch * 255).detach().cpu().numpy()

            input_img = image.copy()
            color = image_vis_batch[0].copy()
            valid_mask = valid_mask_batch[0].cpu().numpy()

            alpha = 0.8
            image_vis = alpha * color[:, :, :3] * valid_mask + (
                1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img
            image_vis = image_vis.astype(np.uint8)

            image_vis = draw_keypoints(image_vis, joints2d[idx])

        return image_vis

    return overlay