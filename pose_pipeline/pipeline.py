
import os
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
        import tempfile
        from pose_pipeline.wrappers.openpose import parse_video, write_video

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
class CenterHMR(dj.Computed):
    definition = '''
    -> Video
    ---
    results : longblob
    output_video      : attach@localattach    # datajoint managed video file
    '''

    def make(self, key):

        from PosePipeline.wrappers.centerhmr import parse_video

        video = (Video & key).fetch1()
        
        res = parse_video(video['video'])

        # don't store verticies or images
        keys_to_keep = ['params',  'pj2d', 'j3d', 'j3d_smpl24', 'j3d_spin24', 'j3d_op25']
        res = [{k: v for k, v in r.items() if k in keys_to_keep} for r in res]
        key['results'] = res
        
        outfile = os.path.splitext(video['video'])
        key['output_video'] = outfile[0] + '_results' + outfile[1]

        self.insert1(key)


