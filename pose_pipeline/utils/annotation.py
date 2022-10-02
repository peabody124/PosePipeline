import numpy as np
import ipywidgets as widgets
from IPython.display import Video as JupyterVideo
from IPython.display import HTML
from IPython.display import display

from pose_pipeline import *


def assign_video(labeling_key, valid, tracks=None):
    """
    Params:
        labeling_key: dictionary indicating the video being visualized to annotate
        valid: boolean indicating a valid person could be identified
        tracks: list matching the numbers associated with the person in video
    """

    # video_subject_id = 0 corresponds to a valid gait lab subject, and
    # video_subject_id = -1 corresponds to a video with bad person detection
    # the keep_tracks fields should correspond to the number(s) overlying the
    # person in the video

    video_key = (TrackingBbox & labeling_key).fetch1('KEY')

    key = video_key.copy()
    if valid:
        key['video_subject_id'] = 0
        key['keep_tracks'] = tracks
    else:
        key['video_subject_id'] = -1
        key['keep_tracks'] = []

    print(f'Inserting {key} into database') #Inserts into database
    PersonBboxValid.insert1(key)


def annotate(key):

    tracking_method_name = (TrackingBboxMethodLookup & key).fetch1('tracking_method_name')
    print(f'Showing: {key}. Method: {tracking_method_name}')

    video = (TrackingBboxVideo & key).fetch1('output_video')

    # get the track IDs present in the video
    tracks = (TrackingBbox & key).fetch1('tracks')
    tracks = np.unique([[t['track_id'] for track in tracks for t in track]]).tolist()

    display(JupyterVideo(video, height=480, html_attributes="controls muted autoplay"))
    practice = widgets.ToggleButtons(
        options= ['Default', *tracks, 'Multiple', 'Skip', 'Absent', 'Invalid'],
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=["Can't Identify Subject", '0', '1', '2', '3', 'Skip']
    )
    display(practice)

    multiple = widgets.Text(
        value='',
        placeholder='',
        description='Multiple',
        disabled=False
    )
    display(multiple)

    def on_click(change):
        value = change['new']
        if value == 'Multiple':
            value = [int(v) for v in multiple.value.split(',')]
            assign_video(key, True, value)
        elif value == 'Skip':
            print('Skipping')
        elif value == 'Absent':
            assign_video(key, True, [])
        elif value == 'Invalid':
            print('Flagging Invalid')
            assign_video(key, False)
        else:
            value = int(value)
            assign_video(key, True, [value])

        os.remove(video)

    practice.observe(on_click, 'value'"")
    ""