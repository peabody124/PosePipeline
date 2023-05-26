import numpy as np
import pandas as pd
from pose_pipeline.utils.keypoint_matching import compute_iou

def compute_temporal_overlap(tracks,unique_ids,return_df=True):
    
    # Get the maximum track ID value
    max_id = max(unique_ids)
    num_tracks = max_id

    # Initialize overlaps table that includes ids from 0 to max_id inclusive
    overlaps = np.zeros((num_tracks+1, num_tracks+1), dtype=int)
    print(overlaps.shape)
    track_id_counts = {}

    # Go through each frame
    for f in tracks:
        # Get a list of the ids present for the current frame
        id_list = [t['track_id'] for t in f if 'track_id' in t]

        # Keep a count of how often each track_id appears 
        for id in id_list:
            # Initialize each id when it appears or add to the running total
            # of how many frames each id appears in
            if id in track_id_counts:
                track_id_counts[id] += 1
            else:
                track_id_counts[id] = 1
        
        # Keep a count of how often each track_id appears with every other track_id
        if len(id_list) == 1:
            track = id_list[0]
            overlaps[track, track] += 1
        else:
            for i in range(len(id_list)):
                for j in range(i+1, len(id_list)): 
                    overlaps[id_list[i], id_list[j]] += 1
                    overlaps[id_list[j], id_list[i]] += 1
            
    if return_df:
        overlaps = pd.DataFrame(overlaps)

        overlaps.index = range(max_id+1)
        overlaps.columns = range(max_id+1)

    return overlaps, track_id_counts

def get_participant_frame_count(tracks,likely_ids):
    # Check how many frames the likely IDs appeared in the video (based on the tracking algo)
    track_ids_per_frame = get_ids_in_frame(tracks)

    likely_ids_set = set(likely_ids)
    participant_in_frame = 0 

    # In each frame, see if a likely ID is present
    for t_ids in track_ids_per_frame:
        id_in_frame = set(t_ids) & likely_ids_set
        
        if len(id_in_frame) > 0:
            participant_in_frame += 1

    return participant_in_frame

def get_likely_ids(detection_by_frame, decoding_by_frame,window_len=25):
    # Get cumulative sums of detections and decodings
    detection_sum = detection_by_frame.cumsum()
    decoding_sum = decoding_by_frame.cumsum()

    # take the sum of the cumulative sums over a sliding window 
    sum_df = detection_sum.diff(periods=window_len) + decoding_sum.diff(periods=window_len)
    # Find the id that is most likely the subject for each frame
    sum_df['max'] = sum_df.idxmax(axis=1)
    likely_ids = [int(id) for id in sum_df['max'].unique() if not np.isnan(id)]
    all_detected_ids = detection_sum.tail(1).to_dict('records')[0]
    all_decoded_ids = decoding_sum.tail(1).to_dict('records')[0]
    
    return likely_ids, all_detected_ids, all_decoded_ids

def compute_splits(unique_ids, all_track_ids, missing_frames_allowed):
    splits = {id:0 for id in unique_ids}
    consecutive_frames_cnt = {id:0 for id in unique_ids}
    missing_frames_cnt = {id:0 for id in unique_ids}
    consecutive_frame_list = {id:[] for id in unique_ids}
    start_index = {id:0 for id in unique_ids}
    stop_index = {id:-1 for id in unique_ids}

    total_tracks = len(all_track_ids)

    # Iterate through the ids present in each frame
    for i,id_list in enumerate(all_track_ids):
        
        # Of all unique IDs present in the video,
        # look at the ones not currently in the frame 
        for missing_id in (unique_ids - set(id_list)):
            # If these IDs have appeared in at least 1 frame previously
            if consecutive_frames_cnt[missing_id] > 0:
                # Set the number of consecutive frames to 0 (since it is 
                # not in the current frame)
                consecutive_frames_cnt[missing_id] = 0
                # Save the indices of the start and end frames that the 
                # id showed up in consecutively
                stop_index[missing_id] = i-1
                consecutive_frame_list[missing_id].append([start_index[missing_id],stop_index[missing_id]])
                # Reset the start index
                start_index[missing_id] = 0
            if stop_index[missing_id] != -1:
                # Increment the 'missing' count for the current ID
                missing_frames_cnt[missing_id] += 1
        
        # Look at the IDs present in the frame
        for present_id in id_list:
            # If the current ID in frame was previously missing for more
            # than 'missing_frames_allowed' add one to the number of splits
            # for that ID
            if missing_frames_cnt[present_id] > missing_frames_allowed:
                splits[present_id] += 1
            
            # If the number of consecutive frames for the current ID is 0
            # then the current index is the new starting index (since it 
            # is now in frame)
            if consecutive_frames_cnt[present_id] == 0:
                start_index[present_id] = i
            
            # Increment the number of consecutive frames the current ID has appeared in
            consecutive_frames_cnt[present_id] += 1
            # Reset the 'missing' count for the current ID
            missing_frames_cnt[present_id] = 0

    # Go through track IDs present at the end of the video and append the first frame of
    # their appearance and the final frame index to the consecutive frame list        
    for track_id in consecutive_frame_list:
        
        if consecutive_frame_list[track_id] == []:
            consecutive_frame_list[track_id].append([start_index[track_id],total_tracks-1])

    return splits, consecutive_frame_list

def compute_cumulative_iou(likely_ids, bboxes_in_frame, all_track_ids):

    num_likely_ids = len(likely_ids)
    
    # Initialize spatial overlaps table that includes ids from 0 to max_id inclusive
    iou_table = np.zeros((num_likely_ids, num_likely_ids))

    # Create a mapping dictionary for easy indexing
    mapping_table = {id: index for index, id in enumerate(likely_ids)}

    likely_ids_set = set(likely_ids)

    for i in range(len(bboxes_in_frame)):
        # Make sure at least 2 of the likely ids are in the frame
        likely_ids_in_frame = list(set(all_track_ids[i]) & likely_ids_set)
        num_likely_ids_in_frame = len(likely_ids_in_frame)
        
        if num_likely_ids_in_frame > 1:
            # Get the bboxes for the likely ids
            for id1 in range(num_likely_ids_in_frame):
                
                # Get the first likely id in frame
                # and the index it should be mapped to for the 
                # output table
                id_idx1 = likely_ids_in_frame[id1]
                mapped_idx1 = mapping_table[id_idx1]
                
                # get the bbox for the first id
                bbox_id1 = bboxes_in_frame[i][id_idx1]

                for id2 in range(id1+1,num_likely_ids_in_frame):
                    # Get the second likely id in frame
                    # and the index it should be mapped to for the 
                    # output table
                    id_idx2 = likely_ids_in_frame[id2]
                    mapped_idx2 = mapping_table[id_idx2]
                    
                    # get the bbox for the second id
                    bbox_id2 = bboxes_in_frame[i][id_idx2]
                    
                    # Calculate the IoU for the 2 current IDs
                    iou = compute_iou(np.array([bbox_id1]),np.array([bbox_id2]))

                    # Update the running sum in the IoU table
                    iou_table[mapped_idx1, mapped_idx2] += iou
                    iou_table[mapped_idx2, mapped_idx1] += iou

    return iou_table


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

def get_unique_ids(ids_by_frame):
    unique_ids = {id for frame in ids_by_frame for id in frame}

    return unique_ids

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

def get_bboxes_in_frame(tracks):

    bboxes_in_frame = [{t['track_id']:t['tlhw'] for t in track_list} for track_list in tracks]

    return bboxes_in_frame


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
