import os
import math
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader

from pose_pipeline import MODEL_DATA_DIR, OpenPosePerson, BlurredVideo, VideoInfo
from pose_pipeline.pipeline import HumorPerson
from pose_pipeline.env import add_path


class TopDownPose_Humor_Dataset(Dataset):
    # based on RGBVideoDataset from HUMOR

    def __init__(self, key, cam_mat=None, seq_len=60, overlap_len=10):
        super(TopDownPose_Humor_Dataset, self).__init__()
        self.key = key

        self.keypoints = (OpenPosePerson & key).fetch1("keypoints").copy()
        height, width = (VideoInfo & key).fetch1("height", "width")

        T = self.keypoints.shape[0]

        self.seq_len = seq_len
        self.overlap_len = overlap_len

        seq_intervals = []
        self.starts = []
        self.ends = []
        if self.seq_len is not None and self.overlap_len is not None:
            num_seqs = math.ceil((T - self.overlap_len) / (self.seq_len - self.overlap_len))
            r = self.seq_len * num_seqs - self.overlap_len * (num_seqs - 1) - T  # number of extra frames we cover
            extra_o = r // (num_seqs - 1)  # we increase the overlap to avoid these as much as possible
            self.overlap_len = self.overlap_len + extra_o

            new_cov = self.seq_len * num_seqs - self.overlap_len * (
                num_seqs - 1
            )  # now compute how many frames are still left to account for
            r = new_cov - T

            # create intervals
            cur_s = 0
            cur_e = cur_s + self.seq_len
            for int_idx in range(num_seqs):
                seq_intervals.append((cur_s, cur_e))
                self.starts.append(cur_s)
                self.ends.append(cur_e)
                cur_overlap = self.overlap_len
                if int_idx < r:
                    cur_overlap += 1  # update to account for final remainder
                cur_s += self.seq_len - cur_overlap
                cur_e = cur_s + self.seq_len
        print(seq_intervals)

        self.DEFAULT_GROUND = np.array([0.0, -1.0, 0.0, -0.5])

        if cam_mat is None:
            DEFAULT_FOCAL_LEN = (1060.531764702488, 1060.3856705041237)
            self.cam_mat = np.array(
                [[DEFAULT_FOCAL_LEN[0], 0.0, width / 2.0], [0.0, DEFAULT_FOCAL_LEN[1], height / 2.0], [0.0, 0.0, 1.0]]
            )
        else:
            self.cam_mat = cam_mat

    def __len__(self):
        return len(self.starts)  # 1 # len(self.keypoints)

    def __getitem__(self, idx):
        obs_data = dict()
        gt_data = dict()

        obs_data["joints2d"] = torch.Tensor(self.keypoints[self.starts[idx] : self.ends[idx]])
        obs_data["seq_interval"] = torch.Tensor(list([self.starts[idx], self.ends[idx]])).to(torch.int)
        obs_data["floor_plane"] = self.DEFAULT_GROUND

        gt_data["cam_matx"] = self.cam_mat

        return obs_data, gt_data


def process_humor(key, return_raw=False):

    keypoints = (OpenPosePerson & key).fetch1("keypoints").copy()
    fps, height, width = (VideoInfo & key).fetch1("fps", "height", "width")

    dataset = TopDownPose_Humor_Dataset(key)

    data_loader = DataLoader(
        dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=lambda _: np.random.seed(),
    )

    device = "cuda"
    NSTAGES = 3

    vposer_location = os.path.join(MODEL_DATA_DIR, "humor/body_models/vposer_v1_0")
    motion_prior_weights = os.path.join(MODEL_DATA_DIR, "humor/checkpoints/humor/best_model.pth")
    gmm_path = os.path.join(MODEL_DATA_DIR, "humor/checkpoints/init_state_prior_gmm/prior_gmm.npz")
    body_model_path = os.path.join(MODEL_DATA_DIR, "humor/body_models/smplh/neutral/model.npz")

    T = keypoints.shape[0]

    with add_path(os.environ["HUMOR_PATH"]):
        from fitting.fitting_utils import load_vposer
        from models.humor_model import HumorModel
        from body_model.body_model import BodyModel
        from utils.torch import load_state
        from fitting.motion_optimizer import MotionOptimizer
        from utils.logging import Logger

        Logger.log = lambda x: None

        # weights for optimization loss terms. taken from `fit_rgb_demo_no_split.cfg`
        loss_weights = {
            "joints2d": [0.001, 0.001, 0.001],
            "joints3d": [0.0, 0.0, 0.0, 0.0],
            "joints3d_rollout": [0.0, 0.0, 0.0, 0.0],
            "verts3d": [0.0, 0.0, 0.0, 0.0],
            "points3d": [0.0, 0.0, 0.0, 0.0],
            "pose_prior": [0.04, 0.04, 0.0],
            "shape_prior": [0.05, 0.05, 0.05],
            "motion_prior": [0.0, 0.0, 0.075],
            "init_motion_prior": [0.0, 0.0, 0.075],
            "joint_consistency": [0.0, 0.0, 100.0],
            "bone_length": [0.0, 0.0, 2000.0],
            "joints3d_smooth": [100.0, 100.0, 0.0],
            "contact_vel": [0.0, 0.0, 100.0],
            "contact_height": [0.0, 0.0, 10.0],
            "floor_reg": [0.0, 0.0, 0.167],
            "rgb_overlap_consist": [200.0, 200.0, 200.0],
        }

        max_loss_weights = {k: max(v) for k, v in loss_weights.items()}
        all_stage_loss_weights = []
        for sidx in range(NSTAGES):
            stage_loss_weights = {k: v[sidx] for k, v in loss_weights.items()}
            all_stage_loss_weights.append(stage_loss_weights)

        use_joints2d = max_loss_weights["joints2d"] > 0.0
        use_overlap_loss = max_loss_weights["rgb_overlap_consist"] > 0.0

        # load vpose body prior
        pose_prior, _ = load_vposer(vposer_location)
        pose_prior = pose_prior.to(device)
        pose_prior.eval()

        # load Humor motion prior
        motion_prior = HumorModel(
            in_rot_rep="mat", out_rot_rep="aa", latent_size=48, model_data_config="smpl+joints+contacts", steps_in=1
        )
        motion_prior.to(device)
        load_state(motion_prior_weights, motion_prior, map_location=device)
        motion_prior.eval()

        # load initial motion distribution
        gmm_res = np.load(gmm_path)
        init_motion_prior = {
            "gmm": (
                torch.Tensor(gmm_res["weights"]).to(device),
                torch.Tensor(gmm_res["means"]).to(device),
                torch.Tensor(gmm_res["covariances"]).to(device),
            )
        }

        results = {
            "trans": [],
            "root_orient": [],
            "pose_body": [],
            "betas": [],
            "latent_pose": [],
            "latent_motion": [],
            "floor_plane": [],
            "contacts": [],
            "vertices": [],
            "seq_interval": [],
        }

        prev_batch_overlap_res_dict = None

        for i, data in enumerate(data_loader):
            observed_data, gt_data = data
            observed_data = {k: v.to(device) for k, v in observed_data.items() if isinstance(v, torch.Tensor)}
            for k, v in gt_data.items():
                if isinstance(v, torch.Tensor):
                    gt_data[k] = v.to(device)

            # pass in the last batch index from previous batch is using overlap consistency
            if use_overlap_loss and prev_batch_overlap_res_dict is not None:
                observed_data["prev_batch_overlap_res"] = prev_batch_overlap_res_dict

            cur_batch_size = observed_data[list(observed_data.keys())[0]].size(0)
            T = observed_data[list(observed_data.keys())[0]].size(1)
            print(f"Cur batch size: {cur_batch_size} and T {T}")

            cam_mat = gt_data["cam_matx"].to(device)

            # load body model
            num_betas = 16
            body_model = BodyModel(
                bm_path=body_model_path,
                num_betas=num_betas,
                batch_size=T * cur_batch_size,
                use_vtx_selector=use_joints2d,
            ).to(device)

            # load optimizer
            robust_loss = "bisquare"
            robust_tuning_const = 4.6851
            joint2d_sigma = 100

            im_dim = (width, height)

            optimizer = MotionOptimizer(
                device,
                body_model,
                num_betas,
                cur_batch_size,
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
                use_chamfer=("points3d" in observed_data),
                im_dim=im_dim,
            )

            optim_result, per_stage_results = optimizer.run(
                observed_data,
                data_fps=fps,
                lr=1.0,
                num_iter=[30, 80, 70],  # 30, 90, 100
                lbfgs_max_iter=50,
                stages_res_out=None,
                fit_gender="neutral",
            )

            # cache results for consistency loss between sequential batches
            if use_overlap_loss:
                prev_batch_overlap_res_dict = dict()
                prev_batch_overlap_res_dict["verts3d"] = per_stage_results["stage3"]["verts3d"][-1].clone().detach()
                prev_batch_overlap_res_dict["betas"] = optim_result["betas"][-1].clone().detach()
                prev_batch_overlap_res_dict["floor_plane"] = optim_result["floor_plane"][-1].clone().detach()
                prev_batch_overlap_res_dict["seq_interval"] = observed_data["seq_interval"][-1].clone().detach()

            for k in optim_result.keys():
                results[k].append(optim_result[k].cpu().detach().numpy())
            results["vertices"].append(per_stage_results["stage3"]["points3d"].cpu().detach().numpy())
            results["seq_interval"].append(observed_data["seq_interval"].cpu().detach().numpy())

            faces = body_model.bm.faces

            if i < (len(data_loader) - 1):
                del optimizer
            del body_model
            del observed_data
            del gt_data
            torch.cuda.empty_cache()

    def stitch_results(results):
        seq_interval = results.pop("seq_interval")

        stitched_results = {k: [] for k in results.keys()}

        last_end = None

        for batch_num, batch_seq in enumerate(seq_interval):

            for i, interval in enumerate(batch_seq):
                if last_end is None:
                    for k in results.keys():
                        stitched_results[k].append(results[k][batch_num][i])
                else:
                    discard = last_end - interval[0]
                    for k in results.keys():
                        if k not in ["betas", "floor_plane"]:
                            stitched_results[k].append(results[k][batch_num][i, discard:])
                        else:
                            stitched_results[k].append(results[k][batch_num][i])

                last_end = interval[-1]

        for k in stitched_results.keys():
            if k not in ["betas", "floor_plane"]:
                stitched_results[k] = np.concatenate(stitched_results[k], axis=0)

        return stitched_results

    stiched_results = stitch_results(results)

    key.update(stiched_results)
    key["faces"] = faces

    if return_raw:
        return key, results
    else:
        return key


def render_humor(key, show_bg=True):
    import tempfile

    out_path = tempfile.mkdtemp()
    print(f"Rendering to {out_path}")

    width, height, fps = (VideoInfo & key).fetch1("width", "height", "fps")
    pose_body, root_orient, trans, beta, contacts, floor_plane = (HumorPerson & key).fetch1(
        "pose_body", "root_orient", "trans", "betas", "contacts", "floor_plane"
    )
    betas = np.tile(np.mean(np.array(beta), axis=0), [pose_body.shape[0], 1])
    floor_plane = np.mean(np.array(floor_plane), axis=0)

    if show_bg:
        import cv2

        imgs = []
        video = (BlurredVideo & key).fetch1("output_video")
        cap = cv2.VideoCapture(video)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            imgs.append(img)
        cap.release()
        os.remove(video)

        img_arr = np.array(imgs) / 256.0
    else:
        img_arr = None

    DEFAULT_GROUND = np.array([0.0, -1.0, 0.0, -0.5])
    DEFAULT_FOCAL_LEN = (1060.531764702488, 1060.3856705041237)
    cam_mat = np.array(
        [[DEFAULT_FOCAL_LEN[0], 0.0, width / 2.0], [0.0, DEFAULT_FOCAL_LEN[1], height / 2.0], [0.0, 0.0, 1.0]]
    )

    # get camera intrinsics
    cam_fx = cam_mat[0, 0]
    cam_fy = cam_mat[1, 1]
    cam_cx = cam_mat[0, 2]
    cam_cy = cam_mat[1, 2]
    cam_intrins = (cam_fx, cam_fy, cam_cx, cam_cy)

    x_frac = float(width) / width
    y_frac = float(height) / height  # if downsampling can use these
    cam_intrins_down = (cam_fx * x_frac, cam_fy * y_frac, cam_cx * x_frac, cam_cy * y_frac)

    body_model_path = os.path.join(MODEL_DATA_DIR, "humor/body_models/smplh/neutral/model.npz")

    os.environ["PYOPENGL_PLATFORM"] = "egl"

    with add_path(os.environ["HUMOR_PATH"]):
        from body_model.body_model import BodyModel
        from viz.utils import viz_smpl_seq
        from viz.utils import create_video

        num_betas = 16
        T = pose_body.shape[0]
        body_model = BodyModel(bm_path=body_model_path, num_betas=num_betas, batch_size=T, use_vtx_selector=True)

        viz = body_model(
            root_orient=torch.Tensor(root_orient),
            pose_body=torch.Tensor(pose_body),
            betas=torch.Tensor(betas),
            trans=torch.Tensor(trans),
        )

        if show_bg:
            BODY_ALPHA = 0.8
        else:
            BODY_ALPHA = 0.5
        SKELETON_ALPHA = 1.0

        viz_smpl_seq(
            viz,
            use_offscreen=True,
            camera_intrinsics=cam_intrins_down,
            contacts=contacts,
            imw=width,
            imh=height,
            fps=fps,
            body_alpha=BODY_ALPHA,
            render_joints=True,
            render_skeleton=SKELETON_ALPHA,
            render_ground=False,
            ground_plane=floor_plane,
            img_seq=img_arr,
            out_path=out_path,
        )

        out_vid = os.path.join(out_path, key["filename"] + "_humor.mp4")
        create_video(os.path.join(out_path, "frame_%08d.png"), out_vid, fps)

    return out_vid
