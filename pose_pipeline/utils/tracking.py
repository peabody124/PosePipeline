import numpy as np
from pose_pipeline import *
import cv2
import pandas as pd
from qreader import QReader


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


class QRdetector:
    def __init__(self):
        pass

    def detect_and_decode(self):
        raise NotImplementedError()


class OpenCVdetector(QRdetector):
    def __init__(self):
        # Creating an instance of the QR detector from OpenCV
        self.qrCodeDetector = cv2.QRCodeDetector()

    def detect_and_decode(self, im):
        cv2Text, cv2points, _ = self.qrCodeDetector.detectAndDecode(im)

        if cv2points is not None:
            # Select the points returned from the QR detector
            cv2points = cv2points[0]

            # get the top left and bottom right corners
            # these points are relative to the image being passed in
            # if the image is cropped, then will need to adjust
            top_left = np.array(cv2points[0]).astype(int)
            bottom_right = np.array(cv2points[2]).astype(int)

            # return the top left and bottom right corners (np.array(x,y))
            # and the text
            return [cv2Text, top_left, bottom_right]

        return False


class QReaderdetector(QRdetector):
    def __init__(self):
        # Creating an instance of the QR detector from OpenCV
        self.qrCodeDetector = QReader()

    def detect_and_decode(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        qreader_out = self.qrCodeDetector.detect_and_decode(image=im, return_bboxes=True)

        if len(qreader_out) > 0:

            # Getting the coordinates for the QR code bbox
            # these points are relative to the image being passed in
            # if the image is cropped, then will need to adjust
            x1, y1, x2, y2 = qreader_out[0][0]
            top_left = np.array([x1, y1]).astype(int)
            bottom_right = np.array([x2, y2]).astype(int)

            qText = qreader_out[0][1]
            if qText == None:
                qText = ""

            # return the top left and bottom right corners (np.array(x,y))
            # and the text
            return [qText, top_left, bottom_right]

        return False


def detect_qr_code(frame, bounding_box,qr_detector):
    """This method attempts to detect a QR code in an image. It takes in an
       image (or video frame) and bounding box to search within. If a QR code
       is detecting, it will return the decoded text and the position of the
       center of the QR code. Otherwise it will return False.

    Parameters
    ----------
    frame: numpy.ndarray
        Numpy array of an image of shape H,W,D

    bounding_box: numpy.ndarray
        Numpy array of the bounding box coordinates wrt to the frame. Bounding
        box coordinates assumed to be in the form [x1,y1,x2,y2]
    """

    frame_copy = frame.copy()
    # Getting shape of input image
    h, w, d = frame_copy.shape

    # Check if any values are negative (cannot find QR code using those indices)
    negative_indices = np.any(bounding_box < 0)

    if negative_indices:
        # print("Cannot find QR code, negative Bbox indices present.")
        return False

    # Extracting bounding box coordinates
    x1 = int(bounding_box[0])
    y1 = int(bounding_box[1])
    x2 = int(bounding_box[2])
    y2 = int(bounding_box[3])

    # Make sure x and y indices are not equal
    if x1 == x2 or y1 == y2:
        # print("Cannot find QR code, Bbox has 0 width or height.")
        return False

    # Check if any values are out of the image bounds
    if y1 > h or y2 > h:
        # print("Cannot find QR code, Bbox indices greater than frame height.")
        return False
    if x1 > w or x2 > w:
        # print("Cannot find QR code, Bbox indices greater than frame width.")
        return False

    # Make sure x1 < x2 and y1 < y2. If not, then swap them
    if x1 > x2:
        # print("Swapping x indices")
        x2, x1 = x1, x2
    if y1 > y2:
        # print("Swapping y indices")
        y2, y1 = y1, y2

    cropped_frame = frame_copy[y1:y2, x1:x2].copy()

    # # create instances of QR code detectors
    # # Just using QReader now but can use OpenCV as well
    # # opencv_detector = OpenCVdetector()
    # qreader_detector = QReaderdetector()

    # Wrap the QR detection in a try/except
    # Sporadic OpenCV errors occur so just skip those frames
    try:
        qr_output = qr_detector.detect_and_decode(cropped_frame)

        if qr_output is not False:

            decodedText, top_left, bottom_right = qr_output
            # Adjust the coordinates to be relative to the overall image rather than the cropped image
            top_left_tuple = tuple([x1, y1] + top_left)
            bottom_right_tuple = tuple([x1, y1] + bottom_right)

            return [decodedText, top_left_tuple, bottom_right_tuple]

    except Exception as e:
        print("Processing error, skipping current frame:")
        print(e)

    return False


def calculate_tracking_confidence(qr_detection_counts, tracks):
    # Total frames
    N = len(tracks)
    Q = sum(qr_detection_counts.values())
    # First determine how often each track_id was present in the video
    track_id_counts = {}
    for track in tracks:
        # print(len(track))
        for id_data in track:
            # print(id_data['track_id'])
            current_id = id_data["track_id"]
            if current_id in track_id_counts:
                track_id_counts[current_id] += 1
            else:
                track_id_counts[current_id] = 0

    # Combine track id counts and qr detection counts into a dataframe
    tracking_info = pd.concat([pd.Series(track_id_counts), pd.Series(qr_detection_counts)], axis=1)
    tracking_info.columns = ["track_id_count", "qr_detection_count"]

    tracking_info["id_in_frame_pct"] = tracking_info["track_id_count"] / N
    tracking_info["pct_of_qr_detections"] = tracking_info["qr_detection_count"] / Q
    tracking_info["confidence"] = tracking_info["id_in_frame_pct"] * tracking_info["pct_of_qr_detections"]

    print(tracking_info)
