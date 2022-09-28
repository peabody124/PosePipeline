"""
Work in progress script to demonstrate ingestion by CBroz1
"""

# Imports
from pose_pipeline.env import set_environmental_variables
from pose_pipeline.paths import get_pose_project_dir
from pose_pipeline import pipeline as p
from datetime import datetime
import datajoint as dj

# Config
dj.config["localattach"]["location"] = "your/local/data/dir"
dj.config["custom"]["pose_project_dir"] = "your/project/data/dir"

# Set env variables
set_environmental_variables()

# Download a video and rename with "%Y%m%d-%H%M%SZ" format
now = datetime.now().strftime("%Y%m%d-%H%M%SZ")
print(now)  # use this to rename dowloaded video in pose_proj_dir

# Insert into Video table
d = p.Video.make_entry(f"{get_pose_project_dir()}/{now}.mp4")
d.update({"video_project": "1"})
p.Video.insert1(d)
p.Video.fetch("KEY")[0]
k = p.Video.fetch("KEY")[0]

# Get Video metadata
p.VideoInfo.populate()
p.VideoInfo()
(p.VideoInfo & "video_project='1'").fetch_timestamps()

#
p.BottomUpMethod.insert1({**k, "bottom_up_method_name": "MMPose"})
p.BottomUpPeople.populate()
## Blocked by not having the checkpoint
## Write func to download when not present?
## https://mmpose.readthedocs.io/_/downloads/en/latest/pdf/
