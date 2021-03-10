import os
import sys

home = os.path.expanduser("~")


def parse_video(video_path, centerhmr_python_path=os.path.join(home, 'projects/pose/CenterHMR/src')):
    sys.path.append(centerhmr_python_path)
    sys.path.append(os.path.join(centerhmr_python_path, 'core'))

    sys.argv = ['demo.py', '--gpu=0', 
            '--gmodel-path=' + os.path.join(centerhmr_python_path, '../trained_models/pw3d_81.8_58.6.pkl'),
            '--configs_yml=' + os.path.join(centerhmr_python_path, 'configs/basic_test_video.yml')]

    from core.test import Demo
    centerhmr_parser = Demo()

    return centerhmr_parser.process_video(video_path)