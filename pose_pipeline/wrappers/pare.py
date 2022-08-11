import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from pose_pipeline import MODEL_DATA_DIR
from pose_pipeline.utils.bounding_box import get_person_dataloader
from pose_pipeline.utils.bounding_box import convert_crop_coords_to_orig_img, convert_crop_cam_to_orig_img
from pose_pipeline.env import add_path
from pose_pipeline import VideoInfo
import torch


def process_pare(key):

    crop_size = 224

    pare_config = os.path.join(MODEL_DATA_DIR, "pare/pare_w_3dpw_config.yaml")
    pare_checkpoint = os.path.join(MODEL_DATA_DIR, "pare/pare_w_3dpw_checkpoint.ckpt")

    with add_path(os.environ["PARE_PATH"]):

        from pare.core.config import get_hparams_defaults, update_hparams, update_hparams_from_dict
        from pare.models import PARE

        model_cfg = update_hparams(pare_config)

        device = "cuda"
        model = PARE(
            backbone=model_cfg.PARE.BACKBONE,
            num_joints=model_cfg.PARE.NUM_JOINTS,
            softmax_temp=model_cfg.PARE.SOFTMAX_TEMP,
            num_features_smpl=model_cfg.PARE.NUM_FEATURES_SMPL,
            focal_length=model_cfg.DATASET.FOCAL_LENGTH,
            img_res=model_cfg.DATASET.IMG_RES,
            pretrained=model_cfg.TRAINING.PRETRAINED,
            iterative_regression=model_cfg.PARE.ITERATIVE_REGRESSION,
            num_iterations=model_cfg.PARE.NUM_ITERATIONS,
            iter_residual=model_cfg.PARE.ITER_RESIDUAL,
            shape_input_type=model_cfg.PARE.SHAPE_INPUT_TYPE,
            pose_input_type=model_cfg.PARE.POSE_INPUT_TYPE,
            pose_mlp_num_layers=model_cfg.PARE.POSE_MLP_NUM_LAYERS,
            shape_mlp_num_layers=model_cfg.PARE.SHAPE_MLP_NUM_LAYERS,
            pose_mlp_hidden_size=model_cfg.PARE.POSE_MLP_HIDDEN_SIZE,
            shape_mlp_hidden_size=model_cfg.PARE.SHAPE_MLP_HIDDEN_SIZE,
            use_keypoint_features_for_smpl_regression=model_cfg.PARE.USE_KEYPOINT_FEATURES_FOR_SMPL_REGRESSION,
            use_heatmaps=model_cfg.DATASET.USE_HEATMAPS,
            use_keypoint_attention=model_cfg.PARE.USE_KEYPOINT_ATTENTION,
            use_postconv_keypoint_attention=model_cfg.PARE.USE_POSTCONV_KEYPOINT_ATTENTION,
            use_scale_keypoint_attention=model_cfg.PARE.USE_SCALE_KEYPOINT_ATTENTION,
            keypoint_attention_act=model_cfg.PARE.KEYPOINT_ATTENTION_ACT,
            use_final_nonlocal=model_cfg.PARE.USE_FINAL_NONLOCAL,
            use_branch_nonlocal=model_cfg.PARE.USE_BRANCH_NONLOCAL,
            use_hmr_regression=model_cfg.PARE.USE_HMR_REGRESSION,
            use_coattention=model_cfg.PARE.USE_COATTENTION,
            num_coattention_iter=model_cfg.PARE.NUM_COATTENTION_ITER,
            coattention_conv=model_cfg.PARE.COATTENTION_CONV,
            use_upsampling=model_cfg.PARE.USE_UPSAMPLING,
            deconv_conv_kernel_size=model_cfg.PARE.DECONV_CONV_KERNEL_SIZE,
            use_soft_attention=model_cfg.PARE.USE_SOFT_ATTENTION,
            num_branch_iteration=model_cfg.PARE.NUM_BRANCH_ITERATION,
            branch_deeper=model_cfg.PARE.BRANCH_DEEPER,
            num_deconv_layers=model_cfg.PARE.NUM_DECONV_LAYERS,
            num_deconv_filters=model_cfg.PARE.NUM_DECONV_FILTERS,
            use_resnet_conv_hrnet=model_cfg.PARE.USE_RESNET_CONV_HRNET,
            use_position_encodings=model_cfg.PARE.USE_POS_ENC,
            use_mean_camshape=model_cfg.PARE.USE_MEAN_CAMSHAPE,
            use_mean_pose=model_cfg.PARE.USE_MEAN_POSE,
            init_xavier=model_cfg.PARE.INIT_XAVIER,
        ).to(device)

        ckpt = torch.load(pare_checkpoint)["state_dict"]
        pretrained_keys = ckpt.keys()
        new_state_dict = OrderedDict()
        for pk in pretrained_keys:
            if pk.startswith("model."):
                new_state_dict[pk.replace("model.", "")] = ckpt[pk]
            else:
                new_state_dict[pk] = ckpt[pk]
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()

        frame_ids, dataloader, bbox = get_person_dataloader(key, crop_size=crop_size)

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

            from scipy.spatial.transform import Rotation as R

            to_rotvec = lambda x: np.array(list(map(lambda y: R.from_matrix(y).as_rotvec(), x))).reshape(-1, 72)

            for batch in tqdm(dataloader):

                batch = batch.to(device)

                batch_size, seqlen = batch.shape[:2]
                output = model(batch)

                pred_cam.append(output["pred_cam"].cpu().detach().numpy())
                pred_verts.append(output["smpl_vertices"].cpu().detach().numpy())
                pred_pose.append(to_rotvec(output["pred_pose"].cpu().detach().numpy()))
                pred_betas.append(output["pred_shape"].cpu().detach().numpy())
                pred_joints3d.append(output["smpl_joints3d"].cpu().detach().numpy())
                smpl_joints2d.append(output["smpl_joints2d"].cpu().detach().numpy())

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
