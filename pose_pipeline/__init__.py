import os

from .pipeline import (schema, Video, VideoInfo, TrackingBboxMethod, TrackingBbox, 
                       BlurredVideo, TrackingBboxVideo, PersonBboxValid, PersonBbox)
from .pipeline import OpenPose, OpenPosePerson, OpenPosePersonVideo
from .pipeline import CenterHMR, CenterHMRPerson, CenterHMRPersonVideo
from .pipeline import ExposePerson, ExposePersonVideo
#from .pipeline import MMPoseTopDownPerson, MMPoseTopDownPersonVideo
#from .pipeline import GastNetPerson, GastNetPersonVideo
#from .pipeline import PoseWarperPerson, PoseWarperPersonVideo
#from .pipeline import PoseFormerPerson
from .pipeline import TopDownMethod, TopDownPerson, TopDownPersonVideo


from .env import add_path, set_environmental_variables

if 'PIPELINE_3RDPARTY' not in os.environ.keys():
    MODEL_DATA_DIR = os.path.join(os.path.split(__file__)[0], '../3rdparty')
else:
    MODEL_DATA_DIR = os.environ['PIPELINE_3RDPARTY']
