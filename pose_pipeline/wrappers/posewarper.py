import os
import cv2
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms

# Note: enforcing this explicitly, because it is fairly easy to compile
# for more recent torch version, but the output simply is invalid
assert torch.__version__ == '1.1.0', "Must use Torch 1.1.0 for PoseWarper"

# PoseWarper isn't well packaged into a module, so expects to have
# the PoseWarper/lib directory in path
import models.pose_hrnet_PoseAgg
from models.pose_hrnet_PoseAgg import get_pose_net
from config import cfg

# Use distribution-aware inference method
from pose_pipeline.utils.inference import get_final_preds
from pose_pipeline.utils.bounding_box import crop_image_bbox

posewarper_dir = os.path.join(os.path.split(models.pose_hrnet_PoseAgg.__file__)[0], '../..')
posewarper_model_path = os.path.join(posewarper_dir,
                                     'PoseWarper_supp_files/pretrained_models/PoseWarper_posetrack18.pth')
posewarper_experiment_file = os.path.join(posewarper_dir,
                                      'experiments/posetrack/hrnet/w48_384x288_adam_lr1e-4_PoseWarper_inference_temporal_pose_aggregation.yaml')

# load the experiment settings
cfg.merge_from_file(posewarper_experiment_file)
cfg.TEST.BLUR_KERNEL = 11  # support for distribution aware inference

# loading this outside the function to avoid recreating the model many times
model = get_pose_net(cfg, False)
model.load_state_dict(torch.load(posewarper_model_path))
model = model.cuda()

def posewarper_track(video, bboxes, present, step=3, pixel_std = 200.0):
    """ Process a video with PoseWarper given provided bboxes 

        Args:
            video (str): path to the file to process
            bboxes (np.array): bounding boxes in TLHW format
            present (np.array): boolean flag if present in frame

        Returns:
            keypoints (np.array): final keypoints (time x 17 x 3)
    """
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def prepare_frame(frame, bbox, transform=transform, target_size=(288, 384)):
        image, bbox = crop_image_bbox(frame, bbox)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
        image = transform(image)

        return image, bbox

    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    aspect_ratio = width / height

    results = []
    frame_buffer = []

    for idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        # warm it up with the same frame
        if len(frame_buffer) == 0:
            frame_buffer = [frame] * (5 * step)

        frame_buffer.append(frame)
        
        center_idx = np.max(idx - step * 2, 0)

        if center_idx >= 0 and present[center_idx]:  # person was found on the frame in center of sequence        
            assert center_idx >= 0
                       
            #if idx % 50 == 0:
            #    print(idx, bboxes[center_idx], aspect_ratio, h, w, int(scale[0] * 200))

            active_frames = [prepare_frame(f, bboxes[center_idx]) for f in frame_buffer[::step]]
            bbox = active_frames[0][1]
            active_frames = [x[0] for x in active_frames]  # strip out the bounding boxes

            # could do some clever code to group these into batches, but skipping for now
            concat_input = torch.cat((active_frames[2], active_frames[1], active_frames[0], active_frames[3], active_frames[4]), 0)[None, ...].cuda()
            outputs = model(concat_input)
            
            results.append({'heatmaps': outputs.cpu().detach().numpy(), 'bbox': bbox, 'frame': center_idx})
        
        frame_buffer = frame_buffer[1:]  # discard oldest frame
        
    # post process the 
    v = {k: [dic[k] for dic in results] for k in results[0]}

    heatmaps = np.concatenate(v['heatmaps'], axis=0)

    keypoints, maxvals = get_final_preds(cfg, heatmaps, v['bbox'])

    keypoints_final = np.zeros((total_frames, 17, 3), dtype=np.float32)

    frames = np.array(v['frame'])
    keypoints_final[frames, :, :2] = keypoints
    keypoints_final[frames, :, 2] = maxvals[:, 0]

    return keypoints_final