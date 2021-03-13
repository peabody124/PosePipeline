
import os
import cv2
import tempfile
import numpy as np
import datajoint as dj
from .utils.keypoint_matching import match_keypoints_to_bbox

dj.config['stores'] = {
    'localattach': {
        'protocol': 'file',
        'location': '/home/jcotton/projects/pose/dj_pose_videos'
    }
}

schema = dj.schema('pose_pipeline')

# TODO 2: Remove interim videos from primary analyses. When possible, they should simply
# be computed on the fly to save storage space (and less things that need to be kept
# secure). When too slow, they ideally _still_ be split off as a separate class that
# can be populated and deleted based on need. Finally, when that would require too
# much manual rework of the upstream code, then still store as separate objects that
# can ultimately be deleted.

# TODO 3: refactor timestamps out of all the classes and just compute it once. breaks joins
# and it is easy to get when needed. We will use the frame index as the primary thing for
# this pipeline, although timestamps obviously become relevant with downstream analyses.

# TODO 4: implement code that blurs all faces and apply that to all interim analysis videos
# by default. As a step towards this, should also refactor video writer into a general
# utility function. May want to precompute a face-blurred image and store that for subsequent
# visualizations, depending on the ultimate speed. This ties into TODO 2.

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


def get_timestamps(d):
    import cv2
    from datetime import timedelta

    cap = cv2.VideoCapture(d['video'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    times = [d['start_time'] + timedelta(0, i / fps) for i in range(frames)]
    return times

       
@schema
class TrackingBbox(dj.Computed):
    definition = '''
    -> Video
    ---
    tracks            : longblob
    timestamps        : longblob
    output_video      : attach@localattach    # datajoint managed video file
    '''

    def make(self, key):
        from pose_pipeline.deep_sort_yolov4.parser import tracking_bounding_boxes

        print(f"Population {key['filename']}")
        d = (Video & key).fetch1()

        _, fname = tempfile.mkstemp(suffix='.mp4')
        tracks = tracking_bounding_boxes(d['video'], fname)

        key['tracks'] = tracks
        key['timestamps'] = get_timestamps(d)
        key['output_video'] = fname

        self.insert1(key)

        # remove the downloaded video to avoid clutter
        os.remove(d['video'])
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
        res['face_keypoints'] = [r['face_keypoints'] for r in res]
        res['hand_keypoints'] = [r['hand_keypoints'] for r in res]
        
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
class OpenPosePerson(dj.Computed):
    definition = '''
    -> PersonBbox
    -> OpenPose
    ---
    keypoints        : longblob
    openpose_ids     : longblob
    output_video      : attach@localattach    # datajoint managed video file
    '''

    def make(self, key):

        # fetch data     
        keypoints = (OpenPose & key).fetch1('keypoints')
        bbox = (PersonBbox & key).fetch1('bbox')

        res = [match_keypoints_to_bbox(bbox[idx], keypoints[idx]) for idx in range(bbox.shape[0])]
        keypoints, openpose_ids = list(zip(*res)) 

        keypoints = np.array(keypoints)
        openpose_ids = np.array(openpose_ids)

        key['keypoints'] = keypoints
        key['openpose_ids'] = openpose_ids
        
        # TODO: this should probably be another object with a lot of this code generalized
        video_filename = (Video & key).fetch1('video')

        cap = cv2.VideoCapture(video_filename)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dsize = (int(w // 2), int(h // 2))

        _, fname = tempfile.mkstemp(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(fname, fourcc, fps, dsize)

        for idx in range(frames):

            def draw_frame(frame, idx=idx, dsize=dsize, thresh=0.25):
                frame = frame.copy()
                cv2.rectangle(frame, 
                            (int(bbox[idx, 0]), int(bbox[idx, 1])), 
                            (int(bbox[idx, 2] + bbox[idx, 0]), int(bbox[idx, 3] + bbox[idx, 1])),
                            (255, 255, 255), 15)
                for i in range(keypoints.shape[1]):
                    if keypoints[idx, i, -1] > thresh:
                        cv2.circle(frame, (int(keypoints[idx, i, 0]), int(keypoints[idx, i, 1])), 15,
                                (0, 0, 255), -1)
                        cv2.circle(frame, (int(keypoints[idx, i, 0]), int(keypoints[idx, i, 1])), 5,
                                (0, 0, 0), -1)
            
                return cv2.resize(frame, dsize=dsize, interpolation=cv2.INTER_CUBIC)

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            outframe = draw_frame(frame)
            out.write(outframe)
        out.release()
        cap.release()

        key['output_video'] = fname
        self.insert1(key)

        os.remove(video_filename)
        os.remove(fname)


@schema
class CenterHMR(dj.Computed):
    definition = '''
    -> Video
    ---
    results           : longblob
    timestamps        : longblob
    output_video      : attach@localattach    # datajoint managed video file
    '''

    def make(self, key):

        from pose_pipeline.wrappers.centerhmr import parse_video

        video = (Video & key).fetch1()
        
        _, out_file_name = tempfile.mkstemp(suffix='.mp4')

        res = parse_video(video['video'], out_file_name)

        # don't store verticies or images
        keys_to_keep = ['params',  'pj2d', 'j3d', 'j3d_smpl24', 'j3d_spin24', 'j3d_op25']
        res = [{k: v for k, v in r.items() if k in keys_to_keep} for r in res]
        key['results'] = res
        key['timestamps'] = get_timestamps(video)
        key['output_video'] = out_file_name

        self.insert1(key)

        os.remove(video['video'])
        os.remove(out_file_name)


@schema
class CenterHMRPerson(dj.Computed):
    definition = '''
    -> PersonBbox
    -> CenterHMR
    ---
    keypoints        : longblob
    poses            : longblob
    betas            : longblob
    cams             : longblob
    global_orients   : longblob
    centerhmr_ids    : longblob
    '''

    def make(self, key):

        # TODO: get video resolution, but wait until it is in database
        def convert_keypoints_to_image(keypoints, imsize=[1080, 1920]):    
            mp = np.array(imsize) * 0.5
            scale = np.max(np.array(imsize)) * 0.5

            keypoints_image = keypoints * scale + mp
            return list(keypoints_image)

        print(key)

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
                                   if id is not None else np.empty((69,)).fill(np.nan) 
                                   for res, id in zip(hmr_results, centerhmr_ids)])
        key['betas'] = np.asarray([res['params']['betas'][id]
                                   if id is not None else np.empty((10,)).fill(np.nan) 
                                   for res, id in zip(hmr_results, centerhmr_ids)])
        key['cams'] = np.asarray([res['params']['cam'][id]
                                  if id is not None else np.empty((3,)).fill(np.nan) 
                                  for res, id in zip(hmr_results, centerhmr_ids)])
        key['global_orients'] = np.asarray([res['params']['global_orient'][id]
                                            if id is not None else np.empty((3,)).fill(np.nan) 
                                            for res, id in zip(hmr_results, centerhmr_ids)])

        key['keypoints'] = np.asarray(keypoints)
        key['centerhmr_ids'] = np.asarray(centerhmr_ids)

        self.insert1(key)


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
    def make_video(key, downsample=4):
        """ Create an overlay video """

        from pose_pipeline.utils.visualization import video_overlay

        video = (BlurredVideo & key).fetch1('output_video')
        keypoints = (PoseWarperPerson & key).fetch1('keypoints')

        def overlay(image, idx, radius=10):
            image = image.copy()
            for i in range(keypoints.shape[1]):
                if keypoints[idx, i, -1] > 0.1:
                    cv2.circle(image, (int(keypoints[idx, i, 0]), int(keypoints[idx, i, 1])), radius, (0, 0, 0), -1)
                    cv2.circle(image, (int(keypoints[idx, i, 0]), int(keypoints[idx, i, 1])), radius-2, (255, 255, 255), -1)
            return image

        _, out_file_name = tempfile.mkstemp(suffix='.mp4')
        video_overlay(video, out_file_name, overlay, downsample=downsample)

        os.remove(video)

        return out_file_name

