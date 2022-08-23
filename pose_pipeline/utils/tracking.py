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


def detect_qr_code(frame, bounding_box):

    frame_copy = frame.copy()
    h, w, d = frame_copy.shape
    # print("frame shape",h,w,d)
    # Check if any values are negative (cannot find QR code using those indices)
    negative_indices = np.any(bounding_box < 0)

    if negative_indices:
        # print("Cannot find QR code, negative Bbox indices present.")
        return False

    x1 = int(bounding_box[0])
    y1 = int(bounding_box[1])
    x2 = int(bounding_box[2])
    y2 = int(bounding_box[3])
    # print("BBOX:", x1, x2, y1, y2)

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

    # w = x2 - x1
    # h = y2 - y1
    # cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0,255,0), 2)
    # print(x1,x2,y1,y2)

    # if x1 < 0 or x2 < 0
    # print("######################################")
    # print(frame_copy.shape)
    # print(f'CROPPING [{y1}:{y2},{x1}:{x2}]')
    frame_to_crop = frame_copy.copy()
    cropped_frame = frame_to_crop[y1:y2, x1:x2].copy()
    # print(x1,x2,y1,y2,w,h,h*w)
    # print("CROPPED: ",cropped_frame.shape)
    # print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    # if w == 0 or h == 0:
    #     print(x1,x2,y1,y2,w,h,h*w)
    #     return False
    # cropped_frame = frame_copy

    qrCodeDetector = cv2.QRCodeDetector()
    decodedText, points, _ = qrCodeDetector.detectAndDecode(cropped_frame)

    if points is not None:
        points = points[0]

        # get center of all points
        local_center = np.mean(np.array(points), axis=0).astype(int)
        global_center = tuple([local_center[0] + x1, local_center[1] + y1])
        if decodedText != "":
            print("DECODED:", decodedText)
        return [decodedText, global_center]

    return False

    # cv2.circle(cropped_frame,center,50,color=(255,0,0),thickness=2)
    # cv2.imshow('frame_to_crop', frame_to_crop)
    # cv2.imshow('image_cropped', cropped_frame)
    #
    # cv2.waitKey(1)

    # if decodedText not in detected_text_list:
    #     detected_text_list.append(decodedText)
    # QR_detected_count += 1
    #
    # if decodedText != '':
    #     decoded_count += 1"
