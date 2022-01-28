import os
import numpy as np
from collections import OrderedDict
from pose_pipeline import MODEL_DATA_DIR, OpenPosePerson, VideoInfo
from pose_pipeline.env import add_path
import torch

def process_humor(key):

    keypoints = (OpenPosePerson & key).fetch1('keypoints').copy()
    fps, height, width = (VideoInfo & key).fetch1('fps', 'height', 'width')

    device = 'cuda'
    NSTAGES = 3
    DEFAULT_GROUND = np.array([0.0, -1.0, 0.0, -0.5])
    DEFAULT_FOCAL_LEN = (1060.531764702488, 1060.3856705041237)
    cam_mat = np.array([[DEFAULT_FOCAL_LEN[0], 0.0, width / 2.],
                        [0.0, DEFAULT_FOCAL_LEN[1], height / 2.],
                        [0.0, 0.0, 1.0]])

    vposer_location = os.path.join(MODEL_DATA_DIR, 'humor/body_models/vposer_v1_0')
    motion_prior_weights = os.path.join(MODEL_DATA_DIR, 'humor/checkpoints/humor/best_model.pth')
    gmm_path = os.path.join(MODEL_DATA_DIR, 'humor/checkpoints/init_state_prior_gmm/prior_gmm.npz')
    body_model_path = os.path.join(MODEL_DATA_DIR, 'humor/body_models/smplh/neutral/model.npz')

    T = keypoints.shape[0]

    observed_data = {'joints2d': torch.Tensor(keypoints[None, ...]).to(device),
                      'floor_plane': torch.Tensor(DEFAULT_GROUND[None, ...]).to(device)}

    with add_path(os.environ['HUMOR_PATH']):
        from fitting.fitting_utils import load_vposer
        from models.humor_model import HumorModel
        from body_model.body_model import BodyModel
        from utils.torch import load_state
        from fitting.motion_optimizer import MotionOptimizer
        from utils.logging import Logger

        Logger.log = lambda x: None

        # weights for optimization loss terms. taken from `fit_rgb_demo_no_split.cfg`
        loss_weights = {
            'joints2d' : [0.001, 0.001, 0.001],
            'joints3d' : [0.0, 0.0, 0.0, 0.0],
            'joints3d_rollout' : [0.0, 0.0, 0.0, 0.0],
            'verts3d' : [0.0, 0.0, 0.0, 0.0],
            'points3d' : [0.0, 0.0, 0.0, 0.0],
            'pose_prior' : [0.04, 0.04, 0.0],
            'shape_prior' : [0.05, 0.05, 0.05],
            'motion_prior' : [0.0, 0.0, 0.075],
            'init_motion_prior' : [0.0, 0.0, 0.075],
            'joint_consistency' : [0.0, 0.0, 100.0],
            'bone_length' : [0.0, 0.0, 2000.0],
            'joints3d_smooth' : [100.0, 100.0, 0.0],
            'contact_vel' : [0.0, 0.0, 100.0],
            'contact_height' : [0.0, 0.0, 10.0],
            'floor_reg' : [0.0, 0.0, 0.167],
            'rgb_overlap_consist' : [200.0, 200.0, 200.0]
        }

        max_loss_weights = {k : max(v) for k, v in loss_weights.items()}
        all_stage_loss_weights = []
        for sidx in range(NSTAGES):
            stage_loss_weights = {k : v[sidx] for k, v in loss_weights.items()}
            all_stage_loss_weights.append(stage_loss_weights)

        use_joints2d = max_loss_weights['joints2d'] > 0.0

        # load vpose body prior
        pose_prior, _ = load_vposer(vposer_location)
        pose_prior = pose_prior.to(device)
        pose_prior.eval()

        # load Humor motion prior
        motion_prior = HumorModel(in_rot_rep='mat',
                                  out_rot_rep='aa',
                                  latent_size=48,
                                  model_data_config='smpl+joints+contacts',
                                  steps_in=1)
        motion_prior.to(device)
        load_state(motion_prior_weights, motion_prior, map_location=device)
        motion_prior.eval()

        # load initial motion distribution
        gmm_res = np.load(gmm_path)
        init_motion_prior = {'gmm': (torch.Tensor(gmm_res['weights']).to(device),
                                    torch.Tensor(gmm_res['means']).to(device),
                                    torch.Tensor(gmm_res['covariances']).to(device))}

        # load body model
        num_betas = 16
        body_model = BodyModel(bm_path=body_model_path, num_betas=num_betas,
                               batch_size=T, use_vtx_selector=use_joints2d).to(device)

        # load optimizer
        robust_loss = 'bisquare'
        robust_tuning_const = 4.6851
        joint2d_sigma = 100

        cam_mat = torch.Tensor(cam_mat[None, ...]).to(device)
        im_dim = (width, height)

        optimizer = MotionOptimizer(device,
                                    body_model,
                                    num_betas,
                                    1,
                                    T,
                                    [k for k in observed_data.keys()],
                                    all_stage_loss_weights,
                                    pose_prior,
                                    motion_prior,
                                    init_motion_prior,
                                    use_joints2d,
                                    cam_mat,
                                    robust_loss,
                                    robust_tuning_const,
                                    joint2d_sigma,
                                    stage3_tune_init_state=True,
                                    stage3_tune_init_num_frames=15,
                                    stage3_tune_init_freeze_start=30,
                                    stage3_tune_init_freeze_end=55,
                                    stage3_contact_refine_only=True,
                                    use_chamfer=('points3d' in observed_data),
                                    im_dim=im_dim)

        optim_result, per_stage_results = optimizer.run(observed_data,
                                                        data_fps=fps,
                                                        lr=1.0,
                                                        num_iter=[30, 80, 70],
                                                        lbfgs_max_iter=50,
                                                        stages_res_out=None,
                                                        fit_gender='neutral')

        for k in optim_result.keys():
            key[k] = optim_result[k][0].cpu().detach().numpy()

        key['vertices'] = per_stage_results['stage3']['points3d'][0].cpu().detach().numpy()
        key['faces'] = body_model.bm.faces

    return key