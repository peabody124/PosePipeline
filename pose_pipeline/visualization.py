import cv2
from tqdm import tqdm


def video_overlay(video, output_name, callback, downsample=4, codec='MP4V'):
    """ Process a video and create overlay image 
    
        Args:
            video (str): filename for source
            output_name (str): output filename
            callback (fn(im, idx) -> im): method to overlay frame
    """

    cap = cv2.VideoCapture(video)

    # get info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # configure output
    output_size = (int(w / downsample), int(h / downsample))
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_name, fourcc, fps, output_size)

    for idx in tqdm(range(total_frames)):

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # process image in RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_frame = callback(frame, idx)

        # move back to BGR format and write to movie
        out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
        out_frame = cv2.resize(out_frame, output_size)
        out.write(out_frame)

    out.release()
    cap.release()
