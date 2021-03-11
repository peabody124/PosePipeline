import numpy as np
import pandas as pd
import json
import datetime
import os
import sys
import cv2

# start datajoint using local server
import datajoint as dj

# if using testing database
dj.config['database.host'] = '127.0.0.1'
dj.config['database.user'] = 'root'
dj.config['database.password'] = 'pose'

dj.config["enable_python_native_blobs"] = True

home = os.path.expanduser("~")

# for using pipeline system
sys.path.append('..')
pipeline_python_path = os.path.join(home, 'projects/pose/PosePipeline')
sys.path.append(pipeline_python_path)
from pose_pipeline.pipeline import VideoSession, Video, CenterHMR, OpenPose

# for openpose to work
openpose_python_path = os.path.join(home, 'projects/pose/openpose/build/python')
sys.path.append(openpose_python_path)

OpenPose.populate()