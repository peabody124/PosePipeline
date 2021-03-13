import os
import sys

# start datajoint using local server
import datajoint as dj

# if using testing database
dj.config['database.host'] = '127.0.0.1'
dj.config['database.user'] = 'root'
dj.config['database.password'] = 'pose'

dj.config["enable_python_native_blobs"] = True


# for PoseWarper to work
home = os.path.expanduser("~")
posewarper_python_path = os.path.join(home, 'projects/pose/PoseWarper')
sys.path.append(os.path.join(posewarper_python_path, 'lib'))

# for using pipeline system
sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))
from pose_pipeline.pipeline import PoseWarperPerson

PoseWarperPerson.populate()