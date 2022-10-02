from pose_pipeline.pipeline import Video
import subprocess
import tempfile
import os


def compress(fn, bitrate=5):
    import subprocess

    fd, temp = tempfile.mkstemp(suffix=".mp4")
    subprocess.run(["ffmpeg", "-y", "-i", fn, "-c:v", "libx264", "-b:v", f"{bitrate}M", temp])
    os.close(fd)
    return temp


def insert_local_video(filename, video_start_time, local_path, video_project="TESTING", skip_duplicates=False):
    """Insert local video into the Pose Pipeline"""

    assert os.path.exists(local_path)

    vid_struct = {
        "video_project": video_project,
        "filename": filename,
        "start_time": video_start_time,
        "video": local_path,
    }

    print(vid_struct)
    Video().insert1(vid_struct, skip_duplicates=skip_duplicates)
