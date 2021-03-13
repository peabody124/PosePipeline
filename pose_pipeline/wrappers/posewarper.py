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
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from models.pose_hrnet_PoseAgg import get_pose_net
from core.inference import get_final_preds
from config import cfg

posewarper_dir = os.path.join(os.path.split(models.pose_hrnet_PoseAgg.__file__)[0], '../..')
posewarper_model_path = os.path.join(posewarper_dir,
                                     'PoseWarper_supp_files/pretrained_models/PoseWarper_posetrack18.pth')
posewarper_experiment_file = os.path.join(posewarper_dir,
                                      'experiments/posetrack/hrnet/w48_384x288_adam_lr1e-4_PoseWarper_inference_temporal_pose_aggregation.yaml')

# load the experiment settings
cfg.merge_from_file(posewarper_experiment_file)

# loading this outside the function to avoid recreating the model many times
model = get_pose_net(cfg, False)
model.load_state_dict(torch.load(posewarper_model_path))
model = model.cuda()

def posewarper_track(video, bboxes, present, step=1, pixel_std = 200.0):
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

    def prepare_frame(frame, center, scale, transform=transform, target_size=(288, 384)):
    
        trans = get_affine_transform(center, scale, 0, target_size)
        image = cv2.warpAffine(frame, trans, target_size, flags=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x = transform(image)

        return x

    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []
    frame_buffer = []

    for idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        frame_buffer.append(frame)
        
        if len(frame_buffer) < (5 * step):  # still warming up
            continue
        
        center_idx = idx - step * 2
        if present[center_idx]:  # person was found on the frame in center of sequence        
            x, y, w, h = bboxes[center_idx]
            center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
            scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
            
            active_frames = [prepare_frame(f, center, scale) for f in frame_buffer[::step]]

            # could do some clever code to group these into batches, but skipping for now
            concat_input = torch.cat((active_frames[2], active_frames[1], active_frames[0], active_frames[3], active_frames[4]), 0)[None, ...].cuda()
            outputs = model(concat_input)
            
            results.append({'heatmaps': outputs.cpu().detach().numpy(), 'center': center, 'scale': scale, 'frame': center_idx})
        
        frame_buffer = frame_buffer[1:]  # discard oldest frame
        
    # post process the 
    v = {k: [dic[k] for dic in results] for k in results[0]}

    heatmaps = np.concatenate(v['heatmaps'], axis=0)

    keypoints, maxvals = get_final_preds(cfg, heatmaps, v['center'], v['scale'])

    keypoints_final = np.zeros((total_frames, 17, 3), dtype=np.float32)

    frames = np.array(v['frame'])
    keypoints_final[frames, :, :2] = keypoints
    keypoints_final[frames, :, 2] = maxvals[:, 0]

    return keypoints_final