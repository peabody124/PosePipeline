from pose_pipeline import *
from pose_pipeline.utils.tracking import annotate_single_person
from typing import List, Dict

def find_lifting_keys(filt=None):
    return ((Video - LiftingPerson) & filt).fetch("KEY")


def top_down_pipeline(key, tracking_method_name="TraDeS", top_down_method_name="MMpose"):
    """
    Run pipeline on a video through to the top down person layer.
    """

    # set up and compute tracking method
    tracking_key = key.copy()
    tracking_method = (TrackingBboxMethodLookup & f'tracking_method_name="{tracking_method_name}"').fetch1(
        "tracking_method"
    )
    tracking_key["tracking_method"] = tracking_method
    TrackingBboxMethod.insert1(tracking_key, skip_duplicates=True)
    TrackingBbox.populate(key, reserve_jobs=True)

    # see if it can be automatically annotated
    annotate_single_person(key)

    # compute the person bbox (requires a method to have inserted the valid bbox)
    PersonBbox.populate(tracking_key, reserve_jobs=True)

    if len(PersonBbox & tracking_key) == 0:
        if len(PersonBboxValid & tracking_key) == 1 and (PersonBboxValid & tracking_key).fetch1("video_subject_id") < 0:
            print(f"Video {key} marked as invalid.")
            return False
        print(f"Waiting for annotation of subject of interest. {tracking_key}")
        return False

    # compute top down person
    top_down_key = (PersonBbox & tracking_key).fetch1("KEY")
    top_down_method = (TopDownMethodLookup & f'top_down_method_name="{top_down_method_name}"').fetch1("top_down_method")
    top_down_key["top_down_method"] = top_down_method
    TopDownMethod.insert1(top_down_key, skip_duplicates=True)
    if top_down_method_name == "OpenPose":
        OpenPose.populate(key)
        OpenPosePerson.populate(key)

    TopDownPerson.populate(top_down_key, reserve_jobs=True)

    # compute some necessary statistics
    VideoInfo.populate(key, reserve_jobs=True)
    DetectedFrames.populate(key, reserve_jobs=True)
    BestDetectedFrames.populate(key, reserve_jobs=True)

    return True


def lifting_pipeline(key, tracking_method_name="TraDeS", top_down_method_name="MMpose", lifting_method_name="GastNet"):
    """
    Run pipeline on a video through to the  lifting layer.
    """

    res = top_down_pipeline(key, tracking_method_name, top_down_method_name)
    if not res:
        return res

    tracking_key = key.copy()
    tracking_method = (TrackingBboxMethodLookup & f'tracking_method_name="{tracking_method_name}"').fetch1(
        "tracking_method"
    )
    tracking_key["tracking_method"] = tracking_method

    top_down_key = (PersonBbox & tracking_key).fetch1("KEY")
    top_down_method = (TopDownMethodLookup & f'top_down_method_name="{top_down_method_name}"').fetch1("top_down_method")
    top_down_key["top_down_method"] = top_down_method

    if len(TopDownPerson & top_down_key) == 0:
        print(f"Top down job must be reserved and not completed. {top_down_key}")
        return False

    # compute lifting
    lifting_key = top_down_key.copy()
    lifting_method = (LiftingMethodLookup & f'lifting_method_name="{lifting_method_name}"').fetch1("lifting_method")
    lifting_key["lifting_method"] = lifting_method
    LiftingMethod.insert1(lifting_key, skip_duplicates=True)
    LiftingPerson.populate(key, reserve_jobs=True)

    if len(LiftingPerson & lifting_key) == 0:
        print(f"Lifting job must be reserved and not completed. {lifting_key}")
        return False

    # compute some necessary statistics
    VideoInfo.populate(key, reserve_jobs=True)
    DetectedFrames.populate(key, reserve_jobs=True)
    BestDetectedFrames.populate(key, reserve_jobs=True)

    return len(LiftingPerson & key) > 0


def bottomup_to_topdown(keys, bottom_up_method_name="OpenPose_BODY25B", tracking_method_name="DeepSortYOLOv4"):
    """
    Compute a BottomUp person and migrate to top down table

    This doesn't stick exactly to DataJoint design patterns, but
    combines a PersonBbox and BottomUp method and then creates a
    TopDownPerson that migrates this data over.

    Params:
        bottom_up_method_name (str) : should match BottomUpMethod and TopDownMethod
        tracking_method_name (str)  : tracking method of PersonBbox to use to identify person

    Returns:
        list of resulting keys
    """

    results = []
    if type(keys) == dict:
        keys = list(keys)

    for key in keys:
        key = key.copy()

        # get this here to confirm it will work below
        bbox_key = (
            PersonBbox & key & (TrackingBboxMethodLookup & {"tracking_method_name": tracking_method_name})
        ).fetch1("KEY")

        if bottom_up_method_name in ["Bridging_COCO_25", "Bridging_bml_movi_87"]:
            from pose_pipeline.pipeline import BottomUpBridging, BottomUpBridgingPerson

            BottomUpBridging.populate(key)
            BottomUpBridgingPerson.populate(bbox_key)
        else:
            # compute bottom up method for this video
            key["bottom_up_method_name"] = bottom_up_method_name
            BottomUpMethod.insert1(key, skip_duplicates=True)
            BottomUpPeople.populate(key)

            # use the desired tracking method to identify the person
            key["tracking_method"] = (TrackingBboxMethodLookup & {"tracking_method_name": tracking_method_name}).fetch1(
                "tracking_method"
            )
            BottomUpPerson.populate(key)

        bbox_key["top_down_method"] = (TopDownMethodLookup & {"top_down_method_name": bottom_up_method_name}).fetch1(
            "top_down_method"
        )
        TopDownMethod.insert1(bbox_key, skip_duplicates=True)
        TopDownPerson.populate(bbox_key)

        results.append((TopDownPerson & bbox_key).fetch1("KEY"))

    return results


def bottom_up_pipeline(keys: List[Dict], bottom_up_method_name: str = "OpenPose_HR", reserve_jobs: bool = True):
    """
    Run bottom up method on a video

    Params:
        keys (list) : list of keys (dict) to run bottom up on
        bottom_up_method_name (str) : should match BottomUpMethod and TopDownMethod
    """

    if type(keys) == dict:
        keys = list(keys)

    for key in keys:
        key = key.copy()

        if bottom_up_method_name in ["Bridging_COCO_25", "Bridging_bml_movi_87", "Bridging_OpenPose"]:
            from pose_pipeline.pipeline import BottomUpBridging
            BottomUpBridging.populate(key, reserve_jobs=reserve_jobs)

            if len(BottomUpBridging & key) == 0:
                print(f"Bottom up job must be reserved and not completed. Skipping {key}")
                continue
            
            # migrate those results to BottomUpPeople
            key = (Video & key).fetch1('KEY')
            key["bottom_up_method_name"] = bottom_up_method_name
            BottomUpMethod.insert1(key, skip_duplicates=True)
            BottomUpPeople.populate(key, reserve_jobs=reserve_jobs)

        else:
            # compute bottom up method for this video
            key["bottom_up_method_name"] = bottom_up_method_name
            BottomUpMethod.insert1(key, skip_duplicates=True)
            BottomUpPeople.populate(key, reserve_jobs=reserve_jobs)
