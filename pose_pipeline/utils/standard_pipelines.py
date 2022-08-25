from pose_pipeline import *
from pose_pipeline.utils.tracking import annotate_single_person


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
        print(f"Waiting for annotation of subject of interest. {tracking_key}")
        return False

    # compute top down person
    top_down_key = (PersonBbox & tracking_key).fetch1("KEY")
    top_down_method = (TopDownMethodLookup & f'top_down_method_name="{top_down_method_name}"').fetch1("top_down_method")
    top_down_key["top_down_method"] = top_down_method
    TopDownMethod.insert1(top_down_key, skip_duplicates=True)
    TopDownPerson.populate(key, reserve_jobs=True)

    # compute some necessary statistics
    VideoInfo.populate(key)
    DetectedFrames.populate(key)
    BestDetectedFrames.populate(key)

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

    return len(LiftingPerson & key) > 0
