
import os
import sys
import cv2
import tempfile
import numpy as np
from datetime import datetime, timedelta
import datajoint as dj
from .utils.keypoint_matching import match_keypoints_to_bbox

dj.config['stores'] = {
    'localattach': {
        'protocol': 'file',
        'location': '/home/jcotton/projects/pose/dj_pose_videos'
    }
}

schema = dj.schema('pose_pipeline')

@schema
class VideoSession(dj.Manual):
    definition = '''
    session_id : smallint auto_increment
    ---
    date : date
    irb  : varchar(50)
    '''


@schema
class Video(dj.Manual):
    definition = '''
    -> VideoSession
    filename   : varchar(50)
    ---
    video      : attach@localattach    # datajoint managed video file
    start_time : timestamp             # time of beginning of video, as accurately as known
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
    fps             : float
    height          : int
    width           : int
    frames          : int
    '''

    def make(self, key):
        
        video, start_time = (Video & key).fetch1('video', 'start_time')

        cap = cv2.VideoCapture(video)
        key['fps'] = fps = cap.get(cv2.CAP_PROP_FPS)
        key['frames'] = frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        key['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        key['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        key['timestamps'] = [start_time + timedelta(0, i / fps) for i in range(frames)]

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
        from pose_pipeline.wrappers.openpose import openpose_parse_video

        d = (Video & key).fetch1()

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
class TrackingBbox(dj.Computed):
    definition = '''
    -> Video
    ---
    tracks            : longblob
    '''

    def make(self, key):
        from pose_pipeline.deep_sort_yolov4.parser import tracking_bounding_boxes

        print(f"Populating {key['filename']}")
        d = (Video & key).fetch1()

        tracks = tracking_bounding_boxes(d['video'])

        key['tracks'] = tracks

        self.insert1(key)

        # remove the downloaded video to avoid clutter
        os.remove(d['video'])


@schema
class TrackingBboxVideo(dj.Computed):
    definition = '''
    -> BlurredVideo
    -> TrackingBbox
    ---
    output_video      : attach@localattach    # datajoint managed video file
    '''

    def make(self, key):

        from pose_pipeline.utils.visualization import video_overlay
        
        def overlay_callback(image, idx):    
            image = image.copy()
            
            for track in tracks[idx]:
                bbox = track['tlbr']
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 6)
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 3)
                
                x = int((bbox[0] + bbox[2]) / 2-150)
                y = int((bbox[3] + bbox[1]) / 2)
                cv2.putText(image, "ID: " + str(track['track_id']), (x, y), 0, 2.0e-3 * image.shape[0], (0, 0, 0), thickness=15)
                cv2.putText(image, "ID: " + str(track['track_id']), (x, y), 0, 2.0e-3 * image.shape[0], (255, 255, 255), thickness=10)

            return image

        video = (BlurredVideo & key).fetch1('output_video')
        tracks = (TrackingBbox & key).fetch1('tracks')

        _, fname = tempfile.mkstemp(suffix='.mp4')
        video_overlay(video, fname, overlay_callback, downsample=4)

        key['output_video'] = fname

        self.insert1(key)

        # remove the downloaded video to avoid clutter
        os.remove(video)
        os.remove(fname)

@schema
class Subject(dj.Manual):
    definition = '''
    subject_id        : varchar(50)
    '''
    # TODO: might want to add more information. Should IRB be here instead of VideoSession?


@schema
class PersonBboxValid(dj.Manual):
    definition = '''
    -> TrackingBbox
    -> Subject
    ---
    keep_tracks       : longblob
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
                    return {'present': True, 'bbox': valid[0]['tlwh']}
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
class CenterHMR(dj.Computed):
    definition = '''
    -> Video
    ---
    results           : longblob
    '''

    def make(self, key):

        from pose_pipeline.wrappers.centerhmr import parse_video

        video = (Video & key).fetch1('video')
        
        _, out_file_name = tempfile.mkstemp(suffix='.mp4')

        res = parse_video(video, out_file_name)

        # don't store verticies or images
        keys_to_keep = ['params',  'pj2d', 'j3d', 'j3d_smpl24', 'j3d_spin24', 'j3d_op25']
        res = [{k: v for k, v in r.items() if k in keys_to_keep} for r in res]
        key['results'] = res

        self.insert1(key)

        # not saving the video in database, just to reduce space requirements
        os.remove(out_file_name)
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


@schema
class CenterHMRPersonVideo(dj.Computed):
    definition = '''
    -> CenterHMRPerson
    -> BlurredVideo
    ---
    output_video      : attach@localattach    # datajoint managed video file
    '''

    def make(self, key):

        import platform
        if 'Ubuntu' in platform.version():
            # In Ubuntu, using osmesa mode for rendering
            os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
            
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
    results        : longblob
    '''

    def make(self, key):

        # need to add this to path before importing the parse function
        sys.path.append(os.environ['EXPOSE_PATH'])
        exp_cfg = os.path.join(os.environ['EXPOSE_PATH'], 'data/conf.yaml')

        from pose_pipeline.wrappers.expose import expose_parse_video

        video = (Video & key).fetch1('video')
        bboxes, present = (PersonBbox & key).fetch1('bbox', 'present')

        key['results'] = expose_parse_video(video, bboxes, present, exp_cfg)

        self.insert1(key)

        os.remove(video)

@schema
class ExposePersonVideo(dj.Computed):
    definition = '''
    -> ExposePerson
    ----
    output_video      : attach@localattach    # datajoint managed video file
    '''

    def make(self, key):

        sys.path.append(os.environ['EXPOSE_PATH'])

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
