import numpy as np
import pandas as pd

def compute_temporal_overlap(tracks,num_tracks):
    
    overlaps = np.zeros((num_tracks, num_tracks), dtype=int)
    track_id_counts = {}

    # Go through each frame
    for f in tracks:
        # Get a list of the ids present for the current frame
        id_list = [t['track_id'] for t in f if 'track_id' in t]

        # Keep a count of how often each track_id appears 
        for id in id_list:
            if id in track_id_counts:
                track_id_counts[id] += 1
            else:
                track_id_counts[id] = 0
        
        # Keep a count of how often each track_id appears with every other track_id
        if len(id_list) == 1:
            track = id_list[0]
            overlaps[track-1, track-1] += 1
        else:
            for i in range(len(id_list)):
                for j in range(i+1, len(id_list)): 
                    overlaps[id_list[i] - 1, id_list[j] - 1] += 1
            
    overlaps_df = pd.DataFrame(overlaps)

    overlaps_df.index = range(1,num_tracks+1)
    overlaps_df.columns = range(1,num_tracks+1)

    return overlaps, track_id_counts


def process_detections(qr_frame_data):

    # Create a copy of the frame data
    frame_df_detections = qr_frame_data.copy()
    # Drop any frames which have no detections
    frame_df_detections.dropna(how='all',inplace=True)

    # Replace all Nans with 0s
    frame_df_detections.fillna(0,inplace=True)
    # Replacing all strings with 1s
    # Any string represents a detection
    for decoded_str in sorted(list(map(list,set(list(map(frozenset,qr_frame_data.values.T))))),key=len, reverse=True)[0]:
        frame_df_detections.replace(to_replace=decoded_str,value=1,inplace=True)

    # get the cumulative sum of detections
    detections_sum = frame_df_detections.cumsum()
    # Reset index to just include frames where there was at least one detection
    # detections_sum.index = np.arange(1,len(detections_sum)+1)
    # Divide the number of detections for each track by the index
    detection_percentage = detections_sum.divide(detections_sum.index,axis=0)

    # detection_percentage['likely_subject'] = detection_percentage
    # detections_sum
    # detection_percentage_diff = detection_percentage.diff() / detection_percentage.index.to_series().diff()

    return frame_df_detections

def process_decodings(qr_frame_data):

    # Create copy of frame data
    frame_df_decoding = qr_frame_data.copy()
    # Drop any frames which have no detections
    frame_df_decoding.dropna(how='all',inplace=True)

    # Get a list of the unique values decoded during the video
    unique_decodings = sorted(list(map(list,set(list(map(frozenset,qr_frame_data.values.T))))),key=len, reverse=True)[0]
    unique_decodings = [x for x in unique_decodings if x == x]

    qr_decoded_sorted = sorted(unique_decodings,key=len)

    # Replace all Nans with 0s
    frame_df_decoding.fillna(0,inplace=True)
    # Replacing empty strings with 0 and all other decoded strings with 1
    frame_df_decoding.replace(to_replace=qr_decoded_sorted[0],value=0,inplace=True)
    if len(qr_decoded_sorted) > 1:
        frame_df_decoding.replace(to_replace=qr_decoded_sorted[1],value=1,inplace=True)

    # get the cumulative sum of correct decoding
    decoding_sum = frame_df_decoding.cumsum()
    # decoding_sum
    # Reset index to just include frames where there was at least one detection
    # decoding_sum.index = np.arange(1,len(decoding_sum)+1)
    # Divide the number of detections for each track by the index
    decoding_percentage = decoding_sum.divide(decoding_sum.index,axis=0)

    return frame_df_decoding    

def get_ids_in_frame(tracks):
    """This method creates a list of the track ids present in each frame of a 
       video. It takes in the tracks information (from TrackingBbox) for a 
       video and will append a list of track ids present in the current
       frame to a larger list. Each index in the larger list corresponds to
       a frame of video.

    Parameters
    ----------
    tracks: list
        List of lists of dictionaries containing track id # and bbox coordinates

    """
    ids_in_frame = [[t['track_id'] for t in track_list] for track_list in tracks]

    return ids_in_frame

def get_detections_in_frame(qr_frame_data):
    """This method creates a list of the qr detections in each frame of a 
       video. It takes in the qr_frame_data information (from TrackingBboxQR)
       for a video and will append a list of the track ids containing a QR code
       in the current frame to a larger list. Each index in the larger list 
       corresponds to a frame of video.

    Parameters
    ----------
    qr_frame_data: list
        List of dictionaries in which the keys are track id #s that had a QR code
        and the values for each key are decoded text of the QR code.

    """
    ids_detected_in_frame = [list(f.keys()) for f in qr_frame_data]

    return ids_detected_in_frame


def compute_classification_metrics(ids_in_frame,detections_in_frame,likely_ids):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    idsw = 0

    total_frames = len(ids_in_frame)

    for f in range(total_frames):
        status = ""
        
        # If this is not empty, the current frame contained one of the likely IDs
        subject_in_frame = set(ids_in_frame[f]) & set(likely_ids)
        # If this is not empty, the qr code detected one of the likely IDs in the current frame
        subject_detected_in_frame = set(detections_in_frame[f]) & set(likely_ids)
        
        # If subject_in_frame and subject_detected_in_frame are not empty, then that is a true positive
        # If subject_in_frame and subject_detected_in_frame are both empty, then that is a true negative
        
        # if subject_in_frame is not empty, but subject_detected_in_frame is, then that is a false negative
        # ie one of the likely tracks is in the frame but not detected by the qr code
        
        # if subject_in_frame is empty but ids_detected_in_frame[f] is not, then that is a false positive
        # ie one of the likely tracks is not in the frame but a qr code is detected
        
        # if subject_in_frame is not empty, and ids_detected_in_frame[f] is not empty but subject_detected_in_frame is empty
        # then that is id swap (but can check iou to verify)
        
        # For the current frame, first check if the subject is in the frame
        if len(subject_in_frame) > 0:
            # at least one of the ids associated with the subject is in frame
            if len(subject_detected_in_frame) > 0:
                # One of the likely ids in frame and detected by the qr code
                status = "true positive"
                tp += 1
            else:
                if len(detections_in_frame[f]) > 0:
                    # at least 1 qr detection in the frame but is not the subject
                    status = "id swap"
                    idsw += 1
                else:
                    # subject in frame but no qr detections
                    status = "false negative"
                    fn += 1
        else:
            # subject id not in the frame according to tracks data
            if len(detections_in_frame[f]) > 0:
                # subject not in frame according to tracks but there are qr detections
                status = "false positive"
                fp += 1
            else:
                # subject id not in frame according to tracks data and no qr detections
                status = "true negative"
                tn += 1

        status = ""


def compute_mota(fn,fp,idsw,gt):
    """This method calculates the Multi-Object Tracking Accuracy (MOTA) for 
       a video. The formula to calculate MOTA is 
             MOTA = 1 - (sum(fn+fp+idsw))/sum(gt)

    Parameters
    ----------
    fn: int
        Number of False Negatives
    fp: int
        Number of False Positives
    idsw: int
        Number of ID Swaps
    gt: int
        Number of Ground Truth Detections

    """

    mota = 1 - (fn + fp + idsw)/gt

    return mota

def compute_precision(tp,fp):

    precision = tp/(tp+fp)

    return precision

def compute_recall(tp,fn):

    recall = tp/(tp+fn)

    return recall
