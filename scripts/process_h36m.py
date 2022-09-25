import pose_pipeline
pose_pipeline.set_environmental_variables()
from pose_pipeline import *
from pose_pipeline.utils.standard_pipelines import top_down_pipeline

pose_pipeline.env.pytorch_memory_limit()
pose_pipeline.env.tensorflow_memory_limit()


VIDEO_PROJECT = "h36m"
keys = (Video & f'video_project="{VIDEO_PROJECT}"').fetch('KEY')

for k in keys:
    #top_down_pipeline(k, top_down_method_name="OpenPose", tracking_method_name='DeepSortYOLOv4')
    top_down_pipeline(k, top_down_method_name="MMPoseHalpe", tracking_method_name='DeepSortYOLOv4')
    #top_down_pipeline(k, top_down_method_name="MMPoseHalpe")