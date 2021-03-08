
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
