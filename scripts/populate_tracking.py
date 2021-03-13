import os
import sys

# start datajoint using local server
import datajoint as dj

# if using testing database
dj.config['database.host'] = '127.0.0.1'
dj.config['database.user'] = 'root'
dj.config['database.password'] = 'pose'

dj.config["enable_python_native_blobs"] = True


# for using pipeline system
pose_pipeline_path = os.path.join(os.path.split(__file__)[0], '..')
sys.path.append(pose_pipeline_path)

from pose_pipeline.pipeline import TrackingBbox 
TrackingBbox.populate()
