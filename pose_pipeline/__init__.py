import os

from .pipeline import Video, VideoInfo
from .pipeline import TrackingBboxMethodLookup, TrackingBboxMethod, TrackingBbox, TrackingBboxVideo
from .pipeline import PersonBboxValid, PersonBbox, BlurredVideo, DetectedFrames, BestDetectedFrames
from .pipeline import OpenPose, OpenPosePerson, OpenPosePersonVideo
from .pipeline import CenterHMR, CenterHMRPerson, CenterHMRPersonVideo
from .pipeline import ExposePerson, ExposePersonVideo
#from .pipeline import MMPoseTopDownPerson, MMPoseTopDownPersonVideo
from .pipeline import GastNetPerson, GastNetPersonVideo
#from .pipeline import PoseWarperPerson, PoseWarperPersonVideo
#from .pipeline import PoseFormerPerson
from .pipeline import TopDownMethodLookup, TopDownMethod, TopDownPerson, TopDownPersonVideo
from .pipeline import LiftingMethodLookup, LiftingMethod, LiftingPerson
from .pipeline import SMPLMethodLookup, SMPLMethod, SMPLPerson, SMPLPersonVideo
from .pipeline import WalkingSegments, WalkingSegmentsVideo


from .env import add_path, set_environmental_variables

if 'PIPELINE_3RDPARTY' not in os.environ.keys():
    MODEL_DATA_DIR = os.path.join(os.path.split(__file__)[0], '../3rdparty')
else:
    MODEL_DATA_DIR = os.environ['PIPELINE_3RDPARTY']
