import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from pose_pipeline import MODEL_DATA_DIR, PersonBbox, Video, SMPLPerson, BlurredVideo
from pose_pipeline.env import add_path
from pose_pipeline.utils.bounding_box import convert_crop_coords_to_orig_img, convert_crop_cam_to_orig_img

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage.transform import estimate_transform, warp, resize, rescale


class PixiePosePipeDataset(Dataset):
    # adopted from https://github.com/YadiraF/PIXIE/blob/master/pixielib/datasets/body_datasets.py

    def __init__(
        self,
        key,
        blurred=False,
        iscrop=True,
        crop_size=224,
        hd_size=1024,
        scale=1.1,
        body_detector="rcnn",
        device="cpu",
    ):

        self.key = key

        self.bbox, self.present = (PersonBbox & key).fetch1("bbox", "present")

        self.frames = []
        self.frame_ids = np.where(self.present)[0]

        if blurred:
            video = (BlurredVideo & key).fetch1("output_video")
        else:
            video = Video.get_robust_reader(key, False)

        cap = cv2.VideoCapture(video)
        while True:
            # should match the length of identified person tracks
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        os.remove(video)

        self.crop_size = crop_size
        self.hd_size = hd_size
        self.scale = scale
        self.iscrop = iscrop

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, index):

        frame_id = self.frame_ids[index]
        image = self.frames[frame_id]
        bbox = self.bbox[frame_id]
        h, w, _ = image.shape

        image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)[None, ...]
        if self.iscrop:
            left = bbox[0]
            right = left + bbox[2]
            top = bbox[1]
            bottom = top + bbox[3]
            old_size = max(right - left, bottom - top)
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size * self.scale)
            src_pts = np.array(
                [
                    [center[0] - size / 2, center[1] - size / 2],
                    [center[0] - size / 2, center[1] + size / 2],
                    [center[0] + size / 2, center[1] - size / 2],
                ]
            )
        else:
            src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])
            left = 0
            right = w - 1
            top = 0
            bottom = h - 1
            bbox = [left, top, right, bottom]

        # crop image
        DST_PTS = np.array([[0, 0], [0, self.crop_size - 1], [self.crop_size - 1, 0]])
        tform = estimate_transform("similarity", src_pts, DST_PTS)
        dst_image = warp(image, tform.inverse, output_shape=(self.crop_size, self.crop_size))
        dst_image = dst_image.transpose(2, 0, 1)
        # hd image
        DST_PTS = np.array([[0, 0], [0, self.hd_size - 1], [self.hd_size - 1, 0]])
        tform_hd = estimate_transform("similarity", src_pts, DST_PTS)
        hd_image = warp(image, tform_hd.inverse, output_shape=(self.hd_size, self.hd_size))
        hd_image = hd_image.transpose(2, 0, 1)
        # crop image
        return {
            "image": torch.tensor(dst_image).float(),
            #'name': imagename,
            #'imagepath': imagepath,
            "image_hd": torch.tensor(hd_image).float(),
            "tform": torch.tensor(tform.params).float(),
            "original_image": torch.tensor(image.transpose(2, 0, 1)).float(),
            "bbox": bbox,
            "size": size,
        }


def process_pixie(key):

    crop_size = 224

    dataset = PixiePosePipeDataset(key)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
    bbox, present = (PersonBbox & key).fetch1("bbox", "present")

    with add_path(os.environ["PIXIE_PATH"]):

        from pixielib.pixie import PIXIE
        from pixielib.utils import util
        from pixielib.utils.config import cfg as pixie_cfg

        device = "cuda"

        pixie_cfg.model.use_tex = False
        pixie = PIXIE(config=pixie_cfg, device=device)

        results = {
            "body_cam": [],
            "global_pose": [],
            "partbody_pose": [],
            "neck_pose": [],
            "shape": [],
            "exp": [],
            "head_pose": [],
            "jaw_pose": [],
            "left_hand_pose": [],
            "left_wrist_pose": [],
            "right_wrist_pose": [],
            "right_hand_pose": [],
            "tex": [],
            "light": [],
        }

        with torch.no_grad():

            for batch in tqdm(dataloader):

                util.move_dict_to_device(batch, device)

                param_dict = pixie.encode({"body": batch})

                body = param_dict["body"]
                for k, v in body.items():
                    results[k].append(v.detach().cpu().numpy())

        for k in body.keys():
            results[k] = np.concatenate(results[k], axis=0)

        smplx = pixie.decode({k: torch.tensor(v).to(device) for k, v in results.items()}, param_type="body")

    key["cams"] = results["body_cam"]
    key["verts"] = smplx["vertices"].cpu().detach().numpy()
    key["poses"] = results
    key["betas"] = results["shape"]
    key["joints3d"] = smplx["smplx_kpt3d"].cpu().detach().numpy()
    key["joints2d"] = smplx["smplx_kpt"].cpu().detach().numpy()

    # height, width = (VideoInfo & key).fetch1('height', 'width')
    # key['cams'] = convert_crop_cam_to_orig_img(key['cams'], bbox, width, height)
    key["joints2d"] = convert_crop_coords_to_orig_img(bbox[present], key["joints2d"], crop_size)

    return key


def get_pixie_callback(key):

    results = (SMPLPerson & key).fetch1("poses")
    dataset = PixiePosePipeDataset(key, blurred=True)
    frame_ids = dataset.frame_ids

    with add_path(os.environ["PIXIE_PATH"]):

        from pixielib.pixie import PIXIE
        from pixielib.visualizer import Visualizer
        from pixielib.utils import util
        from pixielib.utils.config import cfg as pixie_cfg

        device = "cuda"

        pixie_cfg.model.use_tex = False
        pixie = PIXIE(config=pixie_cfg, device=device)

        visualizer = Visualizer(render_size=224, config=pixie_cfg, device=device, rasterizer_type="standard")

        def overlay(image, idx):

            idx = np.where(frame_ids == idx)[0]
            if len(idx) == 1:
                idx = idx[0]
            else:
                return image

            poses = {k: torch.tensor(v[None, idx]).to(device) for k, v in results.items()}
            sample = dataset[idx]
            sample["image"] = sample["image"].unsqueeze(0).to(device)

            opdict = pixie.decode(poses, param_type="body")  # pass through SMPLX

            tform = sample["tform"][None, ...]
            tform = torch.inverse(tform).transpose(1, 2).to(device)
            original_image = sample["original_image"][None, ...] / 256.0

            # should be the same as the passed in image.
            # TODO: need to handled frames with missed bounding boxes
            visualizer.recover_position(opdict, sample, tform, original_image)

            visdict = visualizer.render_results(
                opdict, sample["image_hd"].to(device), overlay=True
            )  # , use_deca=True, moderator_weight=param_dict['moderator_weight'])

            # color_shape_images
            image_out = (
                visdict["shape_images"].detach().cpu().numpy()[0] * 0.5
                + sample["image_hd"].detach().cpu().numpy()[0] * 0.5
            )
            image_out = image_out.transpose([1, 2, 0])
            image_out = np.clip(image_out * 255, 0, 255).astype(np.uint8)

            return image_out

    return overlay
