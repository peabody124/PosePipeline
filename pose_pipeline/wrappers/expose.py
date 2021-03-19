import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as dutils
import functools

from expose.data.build import collate_batch
from expose.data.transforms import build_transforms
from expose.data.utils.bbox import bbox_to_center_scale
from expose.data.targets.image_list import to_image_list
from expose.data.targets import BoundingBox
from expose.models.smplx_net import SMPLXNet
from expose.utils.plot_utils import HDRenderer
from expose.utils.checkpointer import Checkpointer

from loguru import logger


# Modification of expose/data/datasets/image_folder.py to handle a video natively
class VideoWithBoxes(dutils.Dataset):
    def __init__(self,
                 video,
                 bboxes,
                 present,
                 transforms=None,
                 scale_factor=1.2,
                 **kwargs):
        super(VideoWithBoxes, self).__init__()

        # to return with metadata
        self.video_name = os.path.splitext(os.path.split(video)[1])[0]
        
        self.cap = cv2.VideoCapture(video)
        
        # frames with valid bounding box
        self.valid_idx = np.where(present)[0]
            
        self.total_frames = len(self.valid_idx)

        self.transforms = transforms

        self.bboxes = bboxes[self.valid_idx]
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, index):
        
        bbox = self.bboxes[index]

        frame_idx = self.valid_idx[index]
        reads = 1 + frame_idx - int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        for _ in range(reads):
            ret, frame = self.cap.read()
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0

        # bounding boxes are stored in datajoint as TLHW format
        bbox = np.array([bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]])
        #bbox = torch.tensor(bbox).to(device=device)

        target = BoundingBox(bbox, size=img.shape)

        center, scale, bbox_size = bbox_to_center_scale(bbox, dset_scale_factor=self.scale_factor)
        target.add_field('bbox_size', bbox_size)
        target.add_field('orig_bbox_size', bbox_size)
        target.add_field('orig_center', center)
        target.add_field('center', center)
        target.add_field('scale', scale)
        target.add_field('original_bbox', bbox)
        target.add_field('frame_idx', self.valid_idx[index])
        
        target.add_field('fname', f'{self.video_name}_{index:03d}')

        if self.transforms is not None:
            full_img, cropped_image, target = self.transforms(img, target)

        return full_img, cropped_image, target, index


def cpu(tensor):
    return tensor.cpu().detach().numpy()


def expose_parse_video(video, bboxes, present, config_file, device=torch.device('cuda'), batch_size=64):

    logger.remove()

    from expose.config.defaults import get_cfg_defaults
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.is_training = False

    def update_dict(d):
        for k, v in d.items():
            if isinstance(v, str) and v.startswith('data'):
                d[k] = os.path.join(os.environ['EXPOSE_PATH'], v)
            if isinstance(v, dict):
                update_dict(v)
    update_dict(cfg)

    # load model with checkpoint
    model = SMPLXNet(cfg)
    model = model.to(device=device)

    # annoyingly, despite above, still need to change working directory to load model
    pwd = os.getcwd()
    os.chdir(os.environ['EXPOSE_PATH'])

    checkpoint_folder = os.path.join(cfg.output_folder, cfg.checkpoint_folder)
    checkpointer = Checkpointer(model, save_dir=checkpoint_folder, pretrained=cfg.pretrained)
    extra_checkpoint_data = checkpointer.load_checkpoint()

    model = model.eval()

    os.chdir(pwd)

    # prepare data parser
    dataset_cfg = cfg.get('datasets', {})
    body_dsets_cfg = dataset_cfg.get('body', {})
    body_transfs_cfg = body_dsets_cfg.get('transforms', {})

    # must be zero with the code above
    num_workers = 0

    transforms = build_transforms(body_transfs_cfg, is_train=False)
    dataset = VideoWithBoxes(video, bboxes, present, transforms=transforms)
    expose_collate = functools.partial(
        collate_batch, use_shared_memory=num_workers > 0,
        return_full_imgs=True)

    expose_dloader = dutils.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=expose_collate,
        drop_last=False,
        pin_memory=True,
    )

    results = {'bbox_size': [], 'bbox_center': [], 'camera_scale': [], 'camera_transl': [], 
               'initial_params': [], 'final_params': [], 'frames': [], 'faces': []}

    for batch in tqdm(expose_dloader, dynamic_ncols=True):

        full_imgs_list, body_imgs, body_targets = batch
        if full_imgs_list is None:
            continue

        full_imgs = to_image_list(full_imgs_list)
        body_imgs = body_imgs.to(device=device)
        body_targets = [target.to(device) for target in body_targets]
        full_imgs = full_imgs.to(device=device)

        torch.cuda.synchronize()
        model_output = model(body_imgs, body_targets, full_imgs=full_imgs, device=device)
        torch.cuda.synchronize()

        # parse the data to save
        bbox_size = [t.get_field('orig_bbox_size') for t in body_targets]
        bbox_center = [t.get_field('orig_center') for t in body_targets]
        frame_idx = [t.get_field('frame_idx') for t in body_targets]
        camera_parameters = model_output['body']['camera_parameters']
        camera_scale = cpu(camera_parameters['scale'])[:, 0].tolist()
        camera_transl = cpu(camera_parameters['translation']).tolist()

        params = model_output['body']['stage_02']
        initial_params = {k: v.cpu().detach().numpy() for k, v in params.items() if k not in ['faces']}
        initial_params = [dict(zip(initial_params,t)) for t in zip(*initial_params.values())]

        params = model_output['body']['final']
        final_params = {k: v.cpu().detach().numpy() for k, v in params.items() if k not in ['full_pose']}
        final_params = [dict(zip(final_params,t)) for t in zip(*final_params.values())]

        # add to accumulator
        results['faces'] = model_output['body']['stage_02']['faces']
        results['frames'].extend(frame_idx)
        results['bbox_size'].extend(bbox_size)
        results['bbox_center'].extend(bbox_center)
        results['camera_scale'].extend(camera_scale)
        results['camera_transl'].extend(camera_transl)
        results['initial_params'].extend(initial_params)
        results['final_params'].extend(final_params)

    return results


class ExposeVideoWriter:

    def __init__(self, results, body_crop_size=256, focal_length=5000.0):

        self.renderer = HDRenderer(img_size=body_crop_size)
        self.focal_length = focal_length
        self.results = results
        self.faces = results['faces']

        self.camera_scale = results['camera_scale']
        self.camera_transl = results['camera_transl']
        self.bbox_size = results['bbox_size']
        self.bbox_center = results['bbox_center']
        self.frames = results['frames']

    def get_overlay_fn(self):

        def overlay_frame(image, frame_idx):

            idx = [idx for idx, val in enumerate(self.frames) if val == frame_idx]
            if len(idx) != 1:
                return image
            idx = idx[0]

            image = image / 255.0

            vertices = self.results['final_params'][idx]['vertices']
            #vertices = self.results['initial_params'][idx]['vertices']  # these don't capture hands

            z = 2 * self.focal_length / (self.camera_scale[idx] * self.bbox_size[idx])

            transl = [*self.camera_transl[idx], z]

            image = self.renderer(vertices[None, ...],
                    self.faces,
                    focal_length=[self.focal_length],
                    camera_translation=[transl],
                    camera_center=[self.bbox_center[idx]],
                    bg_imgs=[np.transpose(image, [2, 0, 1])],
                    return_with_alpha=False,
                    body_color=[0.4, 0.4, 0.7]
            )

            image = np.transpose(image[0], [1, 2, 0])
            image = (image * 255).astype(np.uint8)

            return image

        return overlay_frame
