import numpy as np
from pose_pipeline import *


def annotate_single_person(filt, subject_id=0, confirm=False):

    keys = ((TrackingBbox & filt & "num_tracks=1") - PersonBboxValid).fetch("KEY")

    if confirm:
        print(f"Found {len(keys)} videos that can be auto-annotated with only one person present. Type Yes to confirm.")
        response = input()
        if response[0].upper() != "Y":
            print("Aborting")
            return

    for k in keys:
        tracks = (TrackingBbox & k).fetch1("tracks")
        track_id = np.unique([[t["track_id"] for t in t2] for t2 in tracks if len(t2) > 0])
        assert len(track_id) == 1, "Found two tracks, should not have"
        k.update({"video_subject_id": subject_id, "keep_tracks": track_id})
        PersonBboxValid.insert1(k)
