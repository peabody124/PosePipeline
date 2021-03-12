
import os
import cv2
import tempfile
import numpy as np
import datajoint as dj

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
class PersonBboxValid(dj.Manual):
    definition = '''
    -> TrackingBbox
    subject_id        : varchar(50)
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
    timestamps        : longblob
    output_video      : attach@localattach    # datajoint managed video file
    '''

    def make(self, key):
        from pose_pipeline.wrappers.openpose import parse_video

        d = (Video & key).fetch1()

        _, fname = tempfile.mkstemp(suffix='.mp4')
        res = parse_video(d['video'], keypoints_only=False, outfile=fname)

        key['keypoints'] = [r['keypoints'] for r in res]
        key['pose_ids'] = [r['pose_ids'] for r in res]
        key['pose_scores'] = [r['pose_scores'] for r in res]
        key['timestamps'] = get_timestamps(d)
        key['output_video'] = fname

        self.insert1(key)

        # remove the downloaded video to avoid clutter
        os.remove(d['video'])
        os.remove(fname)


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

        def keypoints_to_bbox(keypoints, thresh=0.1, min_keypoints=5):
            valid = keypoints[:, -1] > thresh
            keypoints = keypoints[valid, :-1]
            
            if keypoints.shape[0] < min_keypoints:
                return [0.0, 0.0, 0.0, 0.0]
            
            bbox = [np.min(keypoints[:, 0]), np.min(keypoints[:, 1]), np.max(keypoints[:, 0]), np.max(keypoints[:, 1])]
            bbox = bbox[:2] + [bbox[2] - bbox[0], bbox[3] - bbox[1]]
            
            return bbox

        def IoU(box1: np.ndarray, box2: np.ndarray, tlhw=False, epsilon=1e-8):
            """
            calculate intersection over union cover percent
            
                :param box1: box1 with shape (N,4)
                :param box2: box2 with shape (N,4)
                :tlhw: bool if format is tlhw and need to be converted to tlbr
                :return: IoU ratio if intersect, else 0
            """
            point_num = max(box1.shape[0], box2.shape[0])
            b1p1, b1p2, b2p1, b2p2 = box1[:, :2], box1[:, 2:], box2[:, :2], box2[:, 2:]
            
            if tlhw:
                b1p2 = b1p1 + b1p2
                b2p2 = b2p1 + b2p2   

            # mask that eliminates non-intersecting matrices
            base_mat = np.ones(shape=(point_num,)).astype(float)
            base_mat *= np.all(np.greater(b1p2 - b2p1, 0), axis=1)
            base_mat *= np.all(np.greater(b2p2 - b1p1, 0), axis=1)
            
            # epsilon handles case where a bbox has zero size (so let's make that have a IoU=0)
            intersect_area = np.prod(np.minimum(b2p2, b1p2) - np.maximum(b1p1, b2p1), axis=1).astype(float)
            union_area = np.prod(b1p2 - b1p1, axis=1) + np.prod(b2p2 - b2p1, axis=1) - intersect_area + epsilon
            intersect_ratio = intersect_area / union_area
            
            return base_mat * intersect_ratio

        def match_keypoints_to_bbox(bbox: np.ndarray, keypoints_list: list, thresh=0.3, num_keypoints=25):
            """ Finds the best keypoints with an acceptable IoU, if present """
            
            empty_keypoints = np.zeros((num_keypoints, 3))
            
            if len(keypoints_list) == 0:
                return empty_keypoints, None
            
            bbox = np.reshape(bbox, (1, 4))
            iou = IoU(bbox, np.array([keypoints_to_bbox(k) for k in keypoints_list]) )
            idx = np.argmax(iou)
            
            if iou[idx] > thresh:
                return keypoints_list[idx], idx
            
            return empty_keypoints, None

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
                                (255, 255, 255), -1)
            
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


