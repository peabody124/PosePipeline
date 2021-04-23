
import os
import sys
import cv2
import tempfile
import numpy as np
from datetime import datetime, timedelta

import datajoint as dj

from .utils.keypoint_matching import match_keypoints_to_bbox
from .env import add_path

schema = dj.schema('pose_pipeline')

dj.config['stores'] = {
    'localattach': {
        'protocol': 'file',
        'location': '/mnt/08b179d4-cd3b-4ff2-86b5-e7eadb020223/pose_videos/dj'
    }
}


@schema
class Video(dj.Manual):
    definition = '''
    video_project       : varchar(50)
    filename            : varchar(100)
    ---
    video               : attach@localattach    # datajoint managed video file
    start_time          : timestamp             # time of beginning of video, as accurately as known
    '''

    @staticmethod
    def make_entry(filepath, session_id=None):
        from datetime import datetime
        import os
        
        _, fn = os.path.split(filepath)
        date = datetime.strptime(fn[:16], '%Y%m%d-%H%M%SZ')
        d = {'filename': fn, 'video': filepath, 'start_time': date}
        if session_id is not None:
            d.update({'session_id': session_id})
        return d


@schema
class VideoInfo(dj.Computed):
    definition = '''
    -> Video
    ---
    timestamps      : longblob
    delta_time      : longblob
    fps             : float
    height          : int
    width           : int
    num_frames      : int
    '''

    def make(self, key):
        
        video, start_time = (Video & key).fetch1('video', 'start_time')

        cap = cv2.VideoCapture(video)
        key['fps'] = fps = cap.get(cv2.CAP_PROP_FPS)
        key['num_frames'] = frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        key['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        key['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        key['timestamps'] = [start_time + timedelta(0, i / fps) for i in range(frames)]
        key['delta_time'] = [timedelta(0, i / fps).total_seconds() for i in range(frames)]
        self.insert1(key)

        os.remove(video)


@schema
class OpenPose(dj.Computed):
    definition = '''
    -> Video
    ---
    keypoints         : longblob
    pose_ids          : longblob
    pose_scores       : longblob
    face_keypoints    : longblob
    hand_keypoints    : longblob
    '''

    def make(self, key):
        
        d = (Video & key).fetch1()

        with add_path(os.path.join(os.environ['OPENPOSE_PATH'], 'build/python')):
            from pose_pipeline.wrappers.openpose import openpose_parse_video
            res = openpose_parse_video(d['video'])

        key['keypoints'] = [r['keypoints'] for r in res]
        key['pose_ids'] = [r['pose_ids'] for r in res]
        key['pose_scores'] = [r['pose_scores'] for r in res]
        key['hand_keypoints'] = [r['hand_keypoints'] for r in res]
        key['face_keypoints'] = [r['face_keypoints'] for r in res]
        
        self.insert1(key)

        # remove the downloaded video to avoid clutter
        os.remove(d['video'])


@schema
class MMPoseBottomUpPerson(dj.Computed):
    definition = """
    -> Video
    ---
    keypoints          : longblob
    """

    def make(self, key):
        
        from mmpose.apis import init_pose_model, inference_bottom_up_pose_model
        from tqdm import tqdm

        mmpose_files = os.path.join(os.path.split(__file__)[0], '../3rdparty/mmpose/')
        pose_cfg = os.path.join(mmpose_files, 'config/bottom_up/higherhrnet/coco/higher_hrnet48_coco_512x512.py')
        pose_ckpt = os.path.join(mmpose_files, 'checkpoints/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth')

        model = init_pose_model(pose_cfg, pose_ckpt)

        video = (Video & key).fetch1('video')
        cap = cv2.VideoCapture(video)

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        keypoints = []
        for frame_id in tqdm(range(video_length)):

            # should match the length of identified person tracks
            ret, frame = cap.read()
            assert ret and frame is not None
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
            res = inference_bottom_up_pose_model(model, frame)[0]

            kps = np.stack([x['keypoints'] for x in res], axis=0)
            keypoints.append(kps)

        key['keypoints'] = keypoints

        os.remove(video)
        self.insert1(key)


@schema
class BlurredVideo(dj.Computed):
    definition = '''
    -> Video
    -> OpenPose
    ---
    output_video      : attach@localattach    # datajoint managed video file
    '''

    def make(self, key):

        from pose_pipeline.utils.visualization import video_overlay
        
        video, keypoints = (Video * OpenPose & key).fetch1('video', 'keypoints')

        def overlay_callback(image, idx):
            image = image.copy()
            if keypoints[idx] is None:
                return image
                
            found_noses = keypoints[idx][:, 0, -1] > 0.1
            nose_positions = keypoints[idx][found_noses, 0, :2]
            neck_positions = keypoints[idx][found_noses, 1, :2]

            radius = np.linalg.norm(neck_positions - nose_positions, axis=1)
            radius = np.clip(radius, 10, 250)

            for i in range(nose_positions.shape[0]):
                center = (int(nose_positions[i, 0]), int(nose_positions[i, 1]))
                cv2.circle(image, center, int(radius[i]), (255, 255, 255), -1)

            return image

        _, out_file_name = tempfile.mkstemp(suffix='.mp4')
        video_overlay(video, out_file_name, overlay_callback, downsample=1)

        key['output_video'] = out_file_name
        self.insert1(key)

        os.remove(out_file_name)
        os.remove(video)


@schema
class TrackingBboxMethod(dj.Manual):
    definition = '''
    -> Video
    tracking_method   : int
    ---
    '''

@schema
class TrackingBbox(dj.Computed):
    definition = '''
    -> TrackingBboxMethod
    ---
    tracks            : longblob
    num_tracks        : int
    '''

    def make(self, key):

        video = (Video & key).fetch1('video')

        if key['tracking_method'] == 0:
            from pose_pipeline.wrappers.deep_sort_yolov4.parser import tracking_bounding_boxes
            tracks = tracking_bounding_boxes(video)
            key['tracks'] = tracks

        elif key['tracking_method'] == 1:
            from pose_pipeline.wrappers.mmtrack import mmtrack_bounding_boxes
            tracks = mmtrack_bounding_boxes(video)
            key['tracks'] = tracks

        else:
            os.remove(video)
            raise Exception("Unsupported tracking method: {key['tracking_method']")

        track_ids = np.unique([t['track_id'] for track in tracks for t in track])
        key['num_tracks'] = len(track_ids)
        
        self.insert1(key)

        # remove the downloaded video to avoid clutter
        if os.path.exists(video):
            os.remove(video)


@schema
class TrackingBboxVideo(dj.Computed):
    definition = '''
    -> BlurredVideo
    -> TrackingBbox
    ---
    output_video      : attach@localattach    # datajoint managed video file
    '''

    def make(self, key):

        import matplotlib
        from pose_pipeline.utils.visualization import video_overlay

        video = (BlurredVideo & key).fetch1('output_video')
        tracks = (TrackingBbox & key).fetch1('tracks')

        N = len(np.unique([t['track_id'] for track in tracks for t in track]))
        colors = matplotlib.cm.get_cmap('hsv', lut=N)
        
        def overlay_callback(image, idx):    
            image = image.copy()
            
            for track in tracks[idx]:
                c = colors(track['track_id'])
                c = (int(c[0] * 255.0), int(c[1] * 255.0), int(c[2] * 255.0))

                bbox = track['tlbr']
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 6)
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), c, 3)
                
                label = str(track['track_id'])
                textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                x = int((bbox[0] + bbox[2]) / 2 - textsize[0] / 2)
                y = int((bbox[3] + bbox[1]) / 2 + textsize[1] / 2)
                cv2.putText(image, label, (x, y), 0, 2.0e-3 * image.shape[0], (255, 255, 255), thickness=4)
                cv2.putText(image, label, (x, y), 0, 2.0e-3 * image.shape[0], c, thickness=2)

            return image

        _, fname = tempfile.mkstemp(suffix='.mp4')
        video_overlay(video, fname, overlay_callback, downsample=1)

        key['output_video'] = fname

        self.insert1(key)

        # remove the downloaded video to avoid clutter
        os.remove(video)
        os.remove(fname)


@schema
class PersonBboxValid(dj.Manual):
    definition = '''
    -> TrackingBbox
    video_subject_id        : int
    ---
    keep_tracks             : longblob
    '''


@schema
class PersonBbox(dj.Computed):
    definition = '''
    -> PersonBboxValid
    ---
    bbox               : longblob
    present            : longblob
    '''

    def make(self, key):
        
        tracks = (TrackingBbox & key).fetch1('tracks')
        keep_tracks = (PersonBboxValid & key).fetch1('keep_tracks')

        def extract_person_track(tracks):
            
            def process_timestamp(track_timestep):
                valid = [t for t in track_timestep if t['track_id'] in keep_tracks]
                if len(valid) == 1:
                    return {'present': True, 'bbox': valid[0]['tlhw']}
                else:
                    return {'present': False, 'bbox': [0.0, 0.0, 0.0, 0.0]}
                
            return [process_timestamp(t) for t in tracks]

        LD = main_track = extract_person_track(tracks) 
        dict_lists = {k: [dic[k] for dic in LD] for k in LD[0]}

        present = np.array(dict_lists['present'])
       
        key['present'] = np.array(dict_lists['present'])
        key['bbox'] =  np.array(dict_lists['bbox'])

        self.insert1(key)


@schema
class OpenPosePerson(dj.Computed):
    definition = '''
    -> PersonBbox
    -> OpenPose
    ---
    keypoints        : longblob
    hand_keypoints   : longblob
    openpose_ids     : longblob
    '''

    def make(self, key):

        # fetch data     
        keypoints, hand_keypoints = (OpenPose & key).fetch1('keypoints', 'hand_keypoints')
        bbox = (PersonBbox & key).fetch1('bbox')

        res = [match_keypoints_to_bbox(bbox[idx], keypoints[idx]) for idx in range(bbox.shape[0])]
        keypoints, openpose_ids = list(zip(*res)) 

        keypoints = np.array(keypoints)
        openpose_ids = np.array(openpose_ids)

        key['keypoints'] = keypoints
        key['openpose_ids'] = openpose_ids

        key['hand_keypoints'] = []

        for openpose_id, hand_keypoint in zip(openpose_ids, hand_keypoints):
            if openpose_id is None:
                key['hand_keypoints'].append(np.zeros((2, 21, 3)))
            else:
                key['hand_keypoints'].append([hand_keypoint[0][openpose_id], hand_keypoint[1][openpose_id]])
        key['hand_keypoints'] = np.asarray(key['hand_keypoints'])

        self.insert1(key)
        

@schema
class OpenPosePersonVideo(dj.Computed):
    definition = '''
    -> OpenPosePerson
    -> BlurredVideo
    ---
    output_video      : attach@localattach    # datajoint managed video file
    '''

    def make(self, key):

        from pose_pipeline.utils.visualization import video_overlay, draw_keypoints

        # fetch data     
        keypoints, hand_keypoints = (OpenPosePerson & key).fetch1('keypoints', 'hand_keypoints')        
        video_filename = (BlurredVideo & key).fetch1('output_video')

        _, fname = tempfile.mkstemp(suffix='.mp4')
        
        video = (BlurredVideo & key).fetch1('output_video')
        keypoints = (OpenPosePerson & key).fetch1('keypoints')

        def overlay(image, idx):
            image = draw_keypoints(image, keypoints[idx])
            image = draw_keypoints(image, hand_keypoints[idx, 0], threshold=0.02)
            image = draw_keypoints(image, hand_keypoints[idx, 1], threshold=0.02)
            return image

        _, out_file_name = tempfile.mkstemp(suffix='.mp4')
        video_overlay(video, out_file_name, overlay, downsample=4)
        key['output_video'] = out_file_name

        self.insert1(key)

        os.remove(out_file_name)
        os.remove(video)


@schema
class TopDownMethod(dj.Manual):
    definition = '''
    -> PersonBbox
    top_down_method    : int
    '''


@schema
class TopDownPerson(dj.Computed):
    definition = '''
    -> TopDownMethod
    ---
    keypoints          : longblob
    '''

    def make(self, key):

        if key['top_down_method'] == 0:
            from .wrappers.top_down import mmpose_top_down_person
            key['keypoints'] = mmpose_top_down_person(key)

        else:
            raise Exception("Method not implemented")

        self.insert1(key)


@schema
class CenterHMR(dj.Computed):
    definition = '''
    -> Video
    ---
    results           : longblob
    '''

    def make(self, key):

        with add_path([os.path.join(os.environ['CENTERHMR_PATH'], 'src'), 
                       os.path.join(os.environ['CENTERHMR_PATH'], 'src/core')]):
            from pose_pipeline.wrappers.centerhmr import centerhmr_parse_video

            video = (Video & key).fetch1('video')
            res = centerhmr_parse_video(video, os.environ['CENTERHMR_PATH'])

        # don't store verticies or images
        keys_to_keep = ['params',  'pj2d', 'j3d', 'j3d_smpl24', 'j3d_spin24', 'j3d_op25']
        res = [{k: v for k, v in r.items() if k in keys_to_keep} for r in res]
        key['results'] = res

        self.insert1(key)

        # not saving the video in database, just to reduce space requirements
        os.remove(video)


@schema
class CenterHMRPerson(dj.Computed):
    definition = '''
    -> PersonBbox
    -> CenterHMR
    -> VideoInfo
    ---
    keypoints        : longblob
    poses            : longblob
    betas            : longblob
    cams             : longblob
    global_orients   : longblob
    centerhmr_ids    : longblob
    '''

    def make(self, key):

        width, height = (VideoInfo & key).fetch1('width', 'height')

        def convert_keypoints_to_image(keypoints, imsize=[width, height]):    
            mp = np.array(imsize) * 0.5
            scale = np.max(np.array(imsize)) * 0.5

            keypoints_image = keypoints * scale + mp
            return list(keypoints_image)

        # fetch data     
        hmr_results = (CenterHMR & key).fetch1('results')
        bbox = (PersonBbox & key).fetch1('bbox')

        # get the 2D keypoints. note these are scaled from (-0.5, 0.5) assuming a
        # square image (hence convert_keypoints_to_image)
        pj2d = [r['pj2d'] if 'pj2d' in r.keys() else np.zeros((0, 25, 2)) for r in hmr_results]
        all_matches = [match_keypoints_to_bbox(bbox[idx], convert_keypoints_to_image(pj2d[idx]), visible=False)
                       for idx in range(bbox.shape[0])]

        keypoints, centerhmr_ids = list(zip(*all_matches)) 

        key['poses'] = np.asarray([res['params']['body_pose'][id] 
                                   if id is not None else np.array([np.nan] * 69) * np.nan
                                   for res, id in zip(hmr_results, centerhmr_ids)])
        key['betas'] = np.asarray([res['params']['betas'][id]
                                   if id is not None else np.array([np.nan] * 10) * np.nan
                                   for res, id in zip(hmr_results, centerhmr_ids)])
        key['cams'] = np.asarray([res['params']['cam'][id]
                                  if id is not None else np.array([np.nan] * 3) * np.nan
                                  for res, id in zip(hmr_results, centerhmr_ids)])
        key['global_orients'] = np.asarray([res['params']['global_orient'][id]
                                            if id is not None else np.array([np.nan] * 3) * np.nan
                                            for res, id in zip(hmr_results, centerhmr_ids)])

        key['keypoints'] = np.asarray(keypoints)
        key['centerhmr_ids'] = np.asarray(centerhmr_ids)

        self.insert1(key)

    @staticmethod
    def joint_names():
        from smplx.joint_names import JOINT_NAMES
        return JOINT_NAMES[:23]


@schema
class CenterHMRPersonVideo(dj.Computed):
    definition = '''
    -> CenterHMRPerson
    -> BlurredVideo
    ---
    output_video      : attach@localattach    # datajoint managed video file
    '''

    def make(self, key):
            
        from pose_estimation.util.pyrender_renderer import PyrendererRenderer
        from pose_estimation.body_models.smpl import SMPL
        from pose_pipeline.utils.visualization import video_overlay

        # fetch data     
        pose_data = (CenterHMRPerson & key).fetch1()        
        video_filename = (BlurredVideo & key).fetch1('output_video')

        _, fname = tempfile.mkstemp(suffix='.mp4')
        
        video = (BlurredVideo & key).fetch1('output_video')

        smpl = SMPL()

        def overlay(image, idx):
            body_pose = np.concatenate([pose_data['global_orients'][idx], pose_data['poses'][idx]])
            body_beta = pose_data['betas'][idx]

            if np.any(np.isnan(body_pose)):
                return image
            
            h, w = image.shape[:2]
            if overlay.renderer is None:
                overlay.renderer = PyrendererRenderer(smpl.get_faces(), (h, w))

            verts = smpl(body_pose.astype(np.float32)[None, ...], body_beta.astype(np.float32)[None, ...])[0][0]

            cam = [pose_data['cams'][idx][0], *pose_data['cams'][idx][:3]]
            if h > w:
                cam[0] = 1.1 ** cam[0] * (h / w)
                cam[1] = (1.1 ** cam[1])
            else:
                cam[0] = 1.1 ** cam[0]
                cam[1] = (1.1 ** cam[1]) * (w / h)
            
            return overlay.renderer(verts, cam, img=image)
        overlay.renderer = None

        _, out_file_name = tempfile.mkstemp(suffix='.mp4')
        video_overlay(video, out_file_name, overlay, downsample=4)
        key['output_video'] = out_file_name

        self.insert1(key)

        os.remove(out_file_name)
        os.remove(video)


@schema
class PoseWarperPerson(dj.Computed):
    definition = '''
    -> PersonBbox
    ---
    keypoints        : longblob
    '''

    def make(self, key):

        from pose_pipeline.wrappers.posewarper import posewarper_track

        video = (Video & key).fetch1('video')
        bbox, present = (PersonBbox & key).fetch1('bbox', 'present')

        key['keypoints'] = posewarper_track(video, bbox, present)

        self.insert1(key)

        os.remove(video)

@schema
class PoseWarperPersonVideo(dj.Computed):
    definition = '''
    -> PoseWarperPerson
    -> BlurredVideo
    ----
    output_video      : attach@localattach    # datajoint managed video file
    '''

    def make(self, key):
        out_file_name = PoseWarperPersonVideo.make_video(key)
        key['output_video'] = out_file_name
        self.insert1(key)

        os.remove(out_file_name)
    
    @staticmethod
    def make_video(key, downsample=4, thresh=0.1):
        """ Create an overlay video """

        from pose_pipeline.utils.visualization import video_overlay

        video = (BlurredVideo & key).fetch1('output_video')
        keypoints = (PoseWarperPerson & key).fetch1('keypoints')

        def overlay(image, idx, radius=10):
            image = image.copy()
            for i in range(keypoints.shape[1]):
                if keypoints[idx, i, -1] > thresh:
                    cv2.circle(image, (int(keypoints[idx, i, 0]), int(keypoints[idx, i, 1])), radius, (0, 0, 0), -1)
                    cv2.circle(image, (int(keypoints[idx, i, 0]), int(keypoints[idx, i, 1])), radius-2, (255, 255, 255), -1)
            return image

        _, out_file_name = tempfile.mkstemp(suffix='.mp4')
        video_overlay(video, out_file_name, overlay, downsample=downsample)

        os.remove(video)

        return out_file_name

@schema
class ExposePerson(dj.Computed):
    definition = '''
    -> PersonBbox
    ---
    poses          : longblob
    joints         : longblob
    results        : longblob
    '''

    def make(self, key):

        # need to add this to path before importing the parse function
        sys.path.append(os.environ['EXPOSE_PATH'])
        exp_cfg = os.path.join(os.environ['EXPOSE_PATH'], 'data/conf.yaml')

        with add_path(os.environ['EXPOSE_PATH']):
            from pose_pipeline.wrappers.expose import expose_parse_video

            video = (Video & key).fetch1('video')
            bboxes, present = (PersonBbox & key).fetch1('bbox', 'present')

            results = expose_parse_video(video, bboxes, present, exp_cfg)
            key['results'] = results
            key['results'].pop('initial_params')
            key['joints'] = np.asarray([r['joints'] for r in results['final_params']])

            from scipy.spatial.transform import Rotation as R
            key['poses'] = np.asarray([R.from_matrix(r['body_pose']).as_rotvec()
                                      for r in results['final_params']])

        self.insert1(key)

        os.remove(video)

    @staticmethod
    def joint_names():
            from smplx.joint_names import JOINT_NAMES
            return JOINT_NAMES

@schema
class ExposePersonVideo(dj.Computed):
    definition = '''
    -> ExposePerson
    ----
    output_video      : attach@localattach    # datajoint managed video file
    '''

    def make(self, key):

        with add_path(os.environ['EXPOSE_PATH']):
            from pose_pipeline.wrappers.expose import ExposeVideoWriter
            from pose_pipeline.utils.visualization import video_overlay

            # fetch data
            video = (BlurredVideo & key).fetch1('output_video')
            results = (ExposePerson & key).fetch1('results')

            vw = ExposeVideoWriter(results)
            overlay_fn = vw.get_overlay_fn()

            _, out_file_name = tempfile.mkstemp(suffix='.mp4')
            video_overlay(video, out_file_name, overlay_fn, downsample=4)

        key['output_video'] = out_file_name

        self.insert1(key)

        os.remove(out_file_name)
        os.remove(video)


@schema
class TopDownPersonVideo(dj.Computed):
    definition = """
    -> TopDownPerson
    -> BlurredVideo
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """

    def make(self, key):
        
        from pose_pipeline.utils.visualization import video_overlay, draw_keypoints

        video = (BlurredVideo & key).fetch1('output_video')
        keypoints = (TopDownPerson & key).fetch1('keypoints')
        
        def overlay_fn(image, idx):
            image = draw_keypoints(image, keypoints[idx])
            return image

        _, out_file_name = tempfile.mkstemp(suffix='.mp4')
        video_overlay(video, out_file_name, overlay_fn, downsample=4)

        key['output_video'] = out_file_name

        self.insert1(key)

        os.remove(out_file_name)
        os.remove(video)

@schema
class GastNetPerson(dj.Computed):
    definition = """
    -> TopDownPerson
    ---
    keypoints_3d       : longblob
    """

    def make(self, key):

        keypoints = (TopDownPerson & key).fetch1('keypoints')
        height, width = (VideoInfo & key).fetch1('height', 'width')

        gastnet_files = os.path.join(os.path.split(__file__)[0], '../3rdparty/gastnet/')

        with add_path(os.environ["GAST_PATH"]):

            import torch
            from model.gast_net import SpatioTemporalModel, SpatioTemporalModelOptimized1f
            from common.graph_utils import adj_mx_from_skeleton
            from common.skeleton import Skeleton
            from tools.inference import gen_pose
            from tools.preprocess import h36m_coco_format, revise_kpts

            def gast_load_model(rf=27):
                if rf == 27:
                    chk = gastnet_files + '27_frame_model.bin'
                    filters_width = [3, 3, 3]
                    channels = 128
                elif rf == 81:
                    chk = gastnet_files + '81_frame_model.bin'
                    filters_width = [3, 3, 3, 3]
                    channels = 64
                else:
                    raise ValueError('Only support 27 and 81 receptive field models for inference!')
                    
                skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                                    joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                                    joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
                adj = adj_mx_from_skeleton(skeleton)

                model_pos = SpatioTemporalModel(adj, 17, 2, 17, filter_widths=filters_width, channels=channels, dropout=0.05)
                
                checkpoint = torch.load(chk)
                model_pos.load_state_dict(checkpoint['model_pos'])
                
                if torch.cuda.is_available():
                    model_pos = model_pos.cuda()
                model_pos.eval()

                return model_pos

            keypoints_reformat, keypoints_score = keypoints[None, ..., :2], keypoints[None, ..., 2]
            keypoints, scores, valid_frames = h36m_coco_format(keypoints_reformat, keypoints_score)

            re_kpts = revise_kpts(keypoints, scores, valid_frames)
            assert len(re_kpts) == 1

            rf = 27
            model_pos = gast_load_model(rf)

            pad = (rf - 1) // 2  # Padding on each side
            causal_shift = 0

            # Generating 3D poses
            prediction = gen_pose(re_kpts, valid_frames, width, height, model_pos, pad, causal_shift)

        key['keypoints_3d'] = prediction[0]
        self.insert1(key)

    @staticmethod
    def joint_names():
        """ GAST-Net 3D follows the output format of Video3D and uses Human3.6 ordering """
        return ['Hip (root)', 'Right hip', 'Right knee', 'Right foot', 'Left hip', 'Left knee', 'Left foot', 'Spine', 'Thorax', 'Nose', 'Head', 'Left shoulder', 'Left elbow', 'Left wrist', 'Right shoulder', 'Right elbow', 'Right wrist']

@schema
class GastNetPersonVideo(dj.Computed):
    definition = """
    -> GastNetPerson
    -> BlurredVideo
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """
    
    def make(self, key):

        keypoints = (TopDownPerson & key).fetch1('keypoints')
        keypoints_3d = (GastNetPerson & key).fetch1('keypoints_3d').copy()
        blurred_video = (BlurredVideo & key).fetch1('output_video')
        width, height, fps = (VideoInfo & key).fetch1('width', 'height', 'fps')
        _, out_file_name = tempfile.mkstemp(suffix='.mp4')

        with add_path(os.environ["GAST_PATH"]):

            from common.graph_utils import adj_mx_from_skeleton
            from common.skeleton import Skeleton
            from tools.inference import gen_pose
            from tools.preprocess import h36m_coco_format, revise_kpts

            from tools.vis_h36m import render_animation

            skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                                joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                                joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
            adj = adj_mx_from_skeleton(skeleton)

            joints_left, joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
            kps_left, kps_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
            rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
            keypoints_metadata = {'keypoints_symmetry': (joints_left, joints_right), 'layout_name': 'Human3.6M', 'num_joints': 17}

            keypoints_reformat, keypoints_score = keypoints[None, ..., :2], keypoints[None, ..., 2]
            keypoints, scores, valid_frames = h36m_coco_format(keypoints_reformat, keypoints_score)
            re_kpts = revise_kpts(keypoints, scores, valid_frames)
            re_kpts = re_kpts.transpose(1, 0, 2, 3)

            keypoints_3d[:, :, 2] -= np.amin(keypoints_3d[:, :, 2])
            anim_output = {'Reconstruction 1': keypoints_3d}

            render_animation(re_kpts, keypoints_metadata, anim_output, skeleton, fps, 30000, np.array(70., dtype=np.float32),
                            out_file_name, input_video_path=blurred_video, viewport=(width, height), com_reconstrcution=False)

        key['output_video'] = out_file_name
        self.insert1(key)

        os.remove(blurred_video)
        os.remove(out_file_name)

@schema
class PoseFormerPerson(dj.Computed):
    definition = """
    -> TopDownPerson
    ---
    keypoints_3d       : longblob
    """

    def make(self, key):

        keypoints = (TopDownPerson & key).fetch1('keypoints')
        height, width = (VideoInfo & key).fetch1('height', 'width')

        poseformer_files = os.path.join(os.path.split(__file__)[0], '../3rdparty/poseformer/')

        receptive_field=81
        num_joints=17

        def coco_h36m(keypoints):
            # adopted from https://github.com/fabro66/GAST-Net-3DPoseEstimation/blob/97a364affe5cd4f68fab030e0210187333fff25e/tools/mpii_coco_h36m.py#L20
            # MIT License
            
            spple_keypoints = [10, 8, 0, 7]
            h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
            coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        
            temporal = keypoints.shape[0]
            keypoints_h36m = np.zeros_like(keypoints, dtype=np.float32)
            htps_keypoints = np.zeros((temporal, 4, 2), dtype=np.float32)

            # htps_keypoints: head, thorax, pelvis, spine
            htps_keypoints[:, 0, 0] = np.mean(keypoints[:, 1:5, 0], axis=1, dtype=np.float32)
            htps_keypoints[:, 0, 1] = np.sum(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]
            htps_keypoints[:, 1, :] = np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)
            htps_keypoints[:, 1, :] += (keypoints[:, 0, :] - htps_keypoints[:, 1, :]) / 3

            htps_keypoints[:, 2, :] = np.mean(keypoints[:, 11:13, :], axis=1, dtype=np.float32)
            htps_keypoints[:, 3, :] = np.mean(keypoints[:, [5, 6, 11, 12], :], axis=1, dtype=np.float32)

            keypoints_h36m[:, spple_keypoints, :] = htps_keypoints
            keypoints_h36m[:, h36m_coco_order, :] = keypoints[:, coco_order, :]

            keypoints_h36m[:, 9, :] -= (keypoints_h36m[:, 9, :] - np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)) / 4
            keypoints_h36m[:, 7, 0] += 2*(keypoints_h36m[:, 7, 0] - np.mean(keypoints_h36m[:, [0, 8], 0], axis=1, dtype=np.float32))
            keypoints_h36m[:, 8, 1] -= (np.mean(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1])*2/3

            return keypoints_h36m

        # reformat keypoints from coco detection to the input of the lifting
        keypoints = coco_h36m(keypoints[..., :2])
        keypoints = keypoints / np.array([height, width])[None, None, :]

        # reshape into temporal frames. shifted in time as we want to estimate for all
        # time and PoseFormer only produces the central timepoint
        dat = []
        for i in range(keypoints.shape[0]-receptive_field+1):
            dat.append(keypoints[i:i+receptive_field, :, :2])
        dat = np.stack(dat, axis=0)

        with add_path(os.environ["POSEFORMER_PATH"]):

            import torch
            import torch.nn as nn
            from common.model_poseformer import PoseTransformer

            poseformer = PoseTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2, embed_dim_ratio=32, depth=4,
                    num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1)

            poseformer = nn.DataParallel(poseformer)

            poseformer.cuda()
            chk = os.path.join(poseformer_files, 'detected81f.bin')

            checkpoint = torch.load(chk, map_location=lambda storage, loc: storage)
            poseformer.load_state_dict(checkpoint['model_pos'], strict=False)

            kp3d = []
            for idx in range(dat.shape[0]):
                frame = torch.Tensor(dat[None, idx]).cuda()
                kp3d.append(poseformer.forward(frame).cpu().detach().numpy()[:, 0, ...])

            del poseformer
            torch.cuda.empty_cache()

            kp3d = np.concatenate([np.zeros((40, 17, 3)), *kp3d, np.zeros((40, 17, 3))], axis=0)

        key['keypoints_3d'] = kp3d
        self.insert1(key)

    @staticmethod
    def joint_names():
        """ PoseFormer follows the output format of Video3D and uses Human3.6 ordering """
        return ['Hip (root)', 'Right hip', 'Right knee', 'Right foot', 'Left hip', 'Left knee', 'Left foot', 'Spine', 'Thorax', 'Nose', 'Head', 'Left shoulder', 'Left elbow', 'Left wrist', 'Right shoulder', 'Right elbow', 'Right wrist']


def get_person_dataloader(key):

    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    from pose_pipeline.utils.bounding_box import crop_image_bbox

    video, tracks, keep_tracks = (Video * TrackingBbox * PersonBboxValid & key).fetch1('video', 'tracks', 'keep_tracks')

    cap = cv2.VideoCapture(video)

    frames = []
    bboxes = []
    frame_ids = []
    for i, idx in enumerate(range(len(tracks))):
        bbox = [t['tlhw'] for t in tracks[idx] if t['track_id'] in keep_tracks]

        # handle the case where person is not tracked in frame
        if len(bbox) == 0:
            continue

        # should match the length of identified person tracks
        ret, frame = cap.read()
        assert ret and frame is not None

        frames.append(frame)
        bboxes.append(bbox)
        frame_ids.append(idx)


    class Inference(Dataset):
        def __init__(self, frames, bboxes=None, joints2d=None, scale=1.0, crop_size=224):

            self.frames = frames
            self.bboxes = bboxes
            self.joints2d = joints2d
            self.scale = scale
            self.crop_size = crop_size
            self.frames = frames
            self.has_keypoints = True if joints2d is not None else False

            def get_default_transform():
                normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
                return transform

            self.transform = get_default_transform()
            self.norm_joints2d = np.zeros_like(self.joints2d)

            if self.has_keypoints:
                bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
                bboxes[:, 2:] = 150. / bboxes[:, 2:]
                self.bboxes = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 2]]).T

                self.image_file_names = self.image_file_names[time_pt1:time_pt2]
                self.joints2d = joints2d[time_pt1:time_pt2]
                self.frames = frames[time_pt1:time_pt2]

        def __len__(self):
            return len(self.frames)

        def __getitem__(self, idx):

            img = cv2.cvtColor(self.frames[idx], cv2.COLOR_BGR2RGB)
            bbox = self.bboxes[idx][0]
            j2d = self.joints2d[idx] if self.has_keypoints else None

            norm_img = crop_image_bbox(img, bbox, target_size=(self.crop_size, self.crop_size), dilate=self.scale)[0]
            norm_img = self.transform(norm_img)

            if self.has_keypoints:
                return norm_img, kp_2d
            else:
                return norm_img

    dataset = Inference(frames, bboxes)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=16)
    
    return frame_ids, dataloader

@schema
class VIBEPerson(dj.Computed):
    definition = '''
    -> PersonBbox
    ---
    cams            : longblob
    poses           : longblob
    betas           : longblob
    verts           : longblob
    joints3d        : longblob
    joints2d        : longblob
    '''

    def make(self, key):
        
        spin_checkpoint = os.path.join(os.path.split(__file__)[0], '../3rdparty/vibe/spin_model_checkpoint.pth.tar')
        vibe_checkpoint = os.path.join(os.path.split(__file__)[0], '../3rdparty/vibe/vibe_model_w_3dpw.pth.tar')

        with add_path(os.environ['VIBE_PATH']):
            frame_ids, dataloader = get_person_dataloader(key)

            import torch
            from lib.models.vibe import VIBE_Demo

            device = 'cuda'
            has_keypoints = False
            model = VIBE_Demo(
                seqlen=16,
                n_layers=2,
                hidden_size=1024,
                add_linear=True,
                use_residual=True,
                pretrained=spin_checkpoint
            ).to('cuda')

            ckpt = torch.load(vibe_checkpoint)
            ckpt = ckpt['gen_state_dict']
            model.load_state_dict(ckpt, strict=False)
            model.eval()

            with torch.no_grad():
                pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d, norm_joints2d = [], [], [], [], [], [], []

                for batch in dataloader:
                    if has_keypoints:
                        batch, nj2d = batch
                        norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                    batch = batch.unsqueeze(0)
                    batch = batch.to(device)

                    batch_size, seqlen = batch.shape[:2]
                    output = model(batch)[-1]
                    
                    pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1).cpu().detach().numpy())
                    pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3).cpu().detach().numpy())
                    pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1).cpu().detach().numpy())
                    pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1).cpu().detach().numpy())
                    pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3).cpu().detach().numpy())
                    smpl_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2).cpu().detach().numpy())
  
            key['cams'] = np.concatenate(pred_cam, axis=0)
            key['verts'] = np.concatenate(pred_verts, axis=0)
            key['poses'] = np.concatenate(pred_pose, axis=0)
            key['betas'] = np.concatenate(pred_betas, axis=0)
            key['joints3d'] = np.concatenate(pred_joints3d, axis=0)
            key['joints2d'] = np.concatenate(smpl_joints2d, axis=0)

            self.insert1(key)

    @staticmethod
    def joint_names():
            from smplx.joint_names import JOINT_NAMES
            return JOINT_NAMES[:23]

@schema
class MEVAPerson(dj.Computed):
    definition = '''
    -> PersonBbox
    ---
    cams            : longblob
    poses           : longblob
    betas           : longblob
    verts           : longblob
    joints3d        : longblob
    joints2d        : longblob
    '''

    def make(self, key):
        
        config_file = os.path.join(os.path.split(__file__)[0], '../3rdparty/meva/train_meva_2.yml')
        pretrained_model = os.path.join(os.path.split(__file__)[0], '../3rdparty/meva/model_best.pth.tar')

        with add_path(os.environ['MEVA_PATH']):
            frame_ids, dataloader = get_person_dataloader(key)

            import torch
            from meva.lib.meva_model import MEVA, MEVA_demo
            from meva.utils.video_config import update_cfg


            device = 'cuda'
            
            cfg = update_cfg(config_file)
            model = MEVA_demo(
                n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                seqlen=cfg.DATASET.SEQLEN,
                hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
                add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
                bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
                use_residual=cfg.MODEL.TGRU.RESIDUAL,
                cfg = cfg.VAE_CFG,
            ).to(device)
 
            ckpt = torch.load(pretrained_model)
            ckpt = ckpt['gen_state_dict']
            model.load_state_dict(ckpt)
            model.eval()

            with torch.no_grad():
                pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

                for batch in dataloader:
                    
                    batch_image = batch.unsqueeze(0)
                    batch_image = batch_image.to(device)

                    batch_size, seqlen = batch_image.shape[:2]
                    output = model(batch_image)[-1]

                    pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1).cpu().detach().numpy())
                    pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3).cpu().detach().numpy())
                    pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1).cpu().detach().numpy())
                    pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1).cpu().detach().numpy())
                    pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3).cpu().detach().numpy())
                    norm_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2).cpu().detach().numpy())

            key['cams'] = np.concatenate(pred_cam, axis=0)
            key['verts'] = np.concatenate(pred_verts, axis=0)
            key['poses'] = np.concatenate(pred_pose, axis=0)
            key['betas'] = np.concatenate(pred_betas, axis=0)
            key['joints3d'] = np.concatenate(pred_joints3d, axis=0)
            key['joints2d'] = np.concatenate(norm_joints2d, axis=0)

            self.insert1(key)

            video = (Video & key).fetch1('video')
            os.remove(video)

    @staticmethod
    def joint_names():
            from smplx.joint_names import JOINT_NAMES
            return JOINT_NAMES[:23]

    @staticmethod
    def keypoint_names():
            with add_path(os.environ['MEVA_PATH']):
                from meva.lib.smpl import JOINT_NAMES
                return JOINT_NAMES
    