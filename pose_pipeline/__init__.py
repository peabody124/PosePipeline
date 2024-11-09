import os

from .pipeline import Video, VideoInfo
from .pipeline import TrackingBboxMethodLookup, TrackingBboxMethod, TrackingBbox, TrackingBboxVideo
from .pipeline import PersonBboxValid, PersonBbox, BlurredVideo, DetectedFrames, BestDetectedFrames
from .pipeline import OpenPose, OpenPosePerson, OpenPosePersonVideo
from .pipeline import CenterHMR, CenterHMRPerson, CenterHMRPersonVideo

# from .pipeline import MMPoseTopDownPerson, MMPoseTopDownPersonVideo
# from .pipeline import PoseWarperPerson, PoseWarperPersonVideo
# from .pipeline import PoseFormerPerson
from .pipeline import BottomUpMethodLookup, BottomUpMethod, BottomUpPeople, BottomUpPerson, BottomUpVideo
from .pipeline import TopDownMethodLookup, TopDownMethod, TopDownPerson, TopDownPersonVideo
from .pipeline import HandBboxMethodLookup,HandBboxMethod, HandBbox, HandPoseEstimation, HandPoseEstimationMethod, HandPoseEstimationMethodLookup
from .pipeline import LiftingMethodLookup, LiftingMethod, LiftingPerson, LiftingPersonVideo
from .pipeline import SMPLMethodLookup, SMPLMethod, SMPLPerson, SMPLPersonVideo


from .env import add_path, set_environmental_variables, pytorch_memory_limit, tensorflow_memory_limit

if "PIPELINE_3RDPARTY" not in os.environ.keys():
    MODEL_DATA_DIR = os.path.join(os.path.split(__file__)[0], "../3rdparty")
else:
    MODEL_DATA_DIR = os.environ["PIPELINE_3RDPARTY"]
