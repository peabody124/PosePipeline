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
                 scale_factor=1.5,
                 **kwargs):
        super(VideoWithBoxes, self).__init__()

        # to return with metadata
        self.video_name = os.path.splitext(os.path.split(video)[1])[0]
        
        cap = cv2.VideoCapture(video)        
        # preload the entire video into memory
        self.imgs = []
        
        # frames with valid bounding box
        self.valid_idx = np.where(present)[0]
        
        # limited processing for now
        #self.valid_idx = self.valid_idx[:100]
        
        for idx in tqdm(self.valid_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret is False or frame is None:
                break
                
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
                
            self.imgs.append(img)
            
        self.total_frames = len(self.imgs)
        cap.release()

        self.transforms = transforms

        self.bboxes = bboxes[self.valid_idx]
        self.scale_factor = scale_factor

    def __len__(self):
        return self.total_frames

    def __getitem__(self, index):
        
        img = self.imgs[index]
        bbox = self.bboxes[index]
        
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


def expose_parse_video(video, bboxes, present, config_file, device=torch.device('cuda'), num_workers=1, batch_size=64):

    logger.remove()

    from expose.config.defaults import get_cfg_defaults
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.is_training = False

    # load model with checkpoint
    model = SMPLXNet(cfg)
    model = model.to(device=device)

    checkpoint_folder = os.path.join(cfg.output_folder, cfg.checkpoint_folder)
    checkpointer = Checkpointer(model, save_dir=checkpoint_folder, pretrained=cfg.pretrained)
    extra_checkpoint_data = checkpointer.load_checkpoint()

    model = model.eval()

    # prepare data parser
    dataset_cfg = cfg.get('datasets', {})
    body_dsets_cfg = dataset_cfg.get('body', {})
    body_transfs_cfg = body_dsets_cfg.get('transforms', {})

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
        camera_transl = cpu(camera_parameters['translation'])[:, 0].tolist()

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