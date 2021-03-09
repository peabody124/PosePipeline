
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

@schema
class OpenPose(dj.Computed):
    definition = '''
    -> Video
    ---
    keypoints : longblob
    '''

    def make(self, key):

        from PosePipeline.wrappers.openpose import parse_video

        d = (Video & key).fetch1()

        res = parse_video(d['video'])

        key['keypoints'] = res

        self.insert1(key)

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


