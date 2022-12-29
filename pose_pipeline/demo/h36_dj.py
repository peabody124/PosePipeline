"""
PosePipe schema to handle the Human3.6 dataset.

Schema mirrors the organization of the dataset

Subject
  Activity (contains Keypoints 3D)
    ActivityView (contains Keypoints 2D, links to video)

"""

import datajoint as dj
import numpy as np
from pose_pipeline import Video

schema = dj.schema("pose_pipeline")


@schema
class Subject(dj.Manual):
    definition = '''
    subject_id  :  int
    '''


@schema
class Activity(dj.Manual):
    definition = '''
    -> Subject
    activity    :  varchar(50)
    trial       :  int
    ---
    keypoints3d :  longblob
    '''

    @staticmethod
    def joint_names():
        ''' H3.6m has a very odd joint convention.

            metadata.xml has the full listing, a number of the joints are called simply
            'site' and appear to be duplicates. Here I used a consistent name for some
            of the duplicates (e.g. Neck). There are also some markers outside the body
            to provide a reference frame. The thumb and wrist end joints, for example,
            are nothing anatomical but indicate the radial direction from the and and
            distal direction, so provide a full 3D pose of the wrist.
        '''
        return ['Hips',
                'Right Hip', 'Right Knee', 'Right Ankle', 'RightToeBase', 'SiteRightFoot',
                'Left Hip', 'Left Knee', 'Left Ankle', 'LeftToeBase', 'SiteLeftFoot',
                'Hips', 'Spine1', 'Neck', 'Head', 'Head1', 'Neck',
                'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Left Wrist', 'Left Thumb', 'Left Wrist End', 'Left Wrist End', 'Neck',
                'Right Shoulder', 'Right Elbow', 'Right Wrist', 'Right Wrist', 'Right Thumb', 'Right Wrist End', 'Right Wrist End']


@schema
class ActivityView(dj.Manual):
    definition = '''
    -> Activity
    -> Video
    ---
    camera               :  varchar(50)
    keypoints2d          :  longblob
    camera_calibration   :  longblob
    '''


def get_camera(subject, camera):
    # Extract vector of camera parameters. Code is adopted from
    # H36MDatabase.m in the original Human 3.6 code.

    # list of calibration values from h3.6m metadata.xml
    x = '-1.38382776045614 -3.16734304390158 0.413636308164621 1841.10702774543 4955.28462344526 1563.4453958977 -1.3751053014935 -3.16142330949792 0.43359546897471 2030.24045958284 4913.63796569427 1611.6313636891 -1.37882393899288 -3.161671325234 0.427248048577424 2120.16013833049 4927.3896940454 1601.45740899546 -1.37711126488867 -3.1635275001196 0.43614424640396 2108.05645658467 4916.35805859584 1600.44815650564 -1.37617713453938 -3.16256766548869 0.440809992980791 2097.39151027444 4880.94465755369 1605.73247183927 -1.37384065585639 -3.14962908578124 0.415338270470213 1935.45174694685 4950.24580396107 1618.08381523153 -1.3767900857079 -3.1563455676984 0.437827264451112 1974.51295625891 4926.35446379868 1597.83264487845 -1.37729100357321 -3.16179684618815 0.423258143295384 2150.65184102118 4896.16095051376 1611.90470612013 -1.38660153616472 -3.18292551468494 0.441478757753939 2044.45852504166 4935.11727984793 1481.22752752863 -1.3781113489611 -3.16311596375085 0.402183069899382 1968.6409782267 5003.45790523123 1605.95713724922 -1.38253857374534 -3.18032944555565 0.435684449022933 2098.44023766473 4926.55465688236 1500.2785741283 -1.35517119070383 3.13934427144743 2.76026318023792 1761.27853428116 -5078.00659454077 1606.2649598335 -1.3560116169585 3.12772502131602 2.73800506759448 2036.16389517962 -5139.3385621042 1614.10120661275 -1.35263543024478 3.12882528385675 2.7516891353684 2123.31672105987 -5118.71133744668 1613.57176689128 -1.35680541595716 3.13109618263462 2.74030306814628 2092.27851350193 -5135.84853074472 1610.17199274892 -1.3563050564389 3.12931805155389 2.74473744708868 2031.70078497138 -5167.93301206581 1612.92305082439 -1.35273482943409 3.12306622510911 2.73860727855607 1969.80390766216 -5128.73876071113 1632.77883867694 -1.35115947363283 3.12541720944921 2.7424245004417 1937.0584290357 -5119.78981556288 1631.56648087072 -1.35529597940683 3.129067978444 2.7681840012377 2219.9656703358 -5148.45307557174 1613.04401647351 -1.36358353731923 3.1489218957052 2.7695783975645 1990.9596621548 -5123.81055155997 1568.80481574437 -1.35303115989022 3.12880171332043 2.72713189442747 2220.22912867327 -5041.76480051425 1612.21017894355 -1.3670893792209 3.14300994840839 2.74314804826489 2083.18240070009 -4912.17282366308 1561.0785790774 -1.35460453835517 -3.09585375458873 -0.415010681416711 -1846.7776610084 5215.04650469073 1491.97246576518 -1.3574174867254 -3.07852558485507 -0.38991571041581 -1689.70442604909 5178.85392024206 1490.13152959193 -1.35785874492557 -3.08036525681956 -0.393393763004446 -1598.17192178668 5172.67658729266 1487.87569770002 -1.35656698237384 -3.08116345944562 -0.386668372798362 -1606.31611907969 5191.68137642844 1493.60373735271 -1.3562743457011 -3.07982159789068 -0.382865371867181 -1620.59486278793 5171.65873305247 1496.43704696787 -1.36295424416142 -3.07297741832933 -0.405919808477019 -1769.59647655694 5185.36115454991 1476.99340929046 -1.36202017323371 -3.07639612744257 -0.385859472962769 -1741.81111844229 5208.24936307788 1464.82464578145 -1.35633305520994 -3.07903933034276 -0.395738228536372 -1571.22149997798 5137.01858150498 1498.17612798791 -1.34888172298336 -3.102960692595 -0.382788963877425 -1670.99215489414 5211.98574196124 1528.38799771705 -1.35664059764913 -3.08015372871798 -0.419008729038864 -1762.35984904783 5158.93496663761 1496.30302817594 -1.33953525239742 -3.10064240051082 -0.434419468030056 -1609.81534310074 5177.33597262935 1537.89671533582 -1.29084725708861 -3.20353825281863 -2.72857298700043 -1794.78972871109 -3722.69891503676 1574.89272604599 -1.31323216561859 -3.22724581097479 -2.75817510960254 -1641.69705160757 -3863.42077254474 1543.98806588784 -1.31327740720992 -3.22661133378847 -2.75939529010902 -1525.62265953577 -3867.76578824123 1550.84956330304 -1.31252617375512 -3.22455904685998 -2.75806537234578 -1585.26651928664 -3848.7127782811 1549.93473021857 -1.31332016240395 -3.22574515339158 -2.75330005891557 -1637.17374540766 -3867.31734917197 1547.03325638793 -1.31574861632979 -3.23364633451714 -2.74452670528462 -1721.66874978818 -3884.13134701978 1540.48790236846 -1.31229903526751 -3.23226714486797 -2.75617673343893 -1734.71057764601 -3832.42135394085 1548.58303462156 -1.31261466850986 -3.22669130539521 -2.76803843215725 -1476.91338239988 -3896.7411238582 1547.97220703876 -1.30216364078597 -3.20525794988117 -2.73209048976503 -1696.04347097168 -3827.09988628541 1591.41272727883 -1.31370197737602 -3.22564758156147 -2.79582077466808 -1459.15110355738 -3879.80915963968 1551.10046698856 -1.3052820100275 -3.20681865219712 -2.73759301328735 -1590.73799072229 -3854.16900368964 1578.01760714384 1145.04940458804 1143.78109572365 512.541504956548 515.4514869776 -0.207098910824901 0.247775183068982 -0.00307515035078854 -0.00142447157470321 -0.000975698859470499 1149.67569986785 1147.59161666764 508.848621645943 508.064917088557 -0.194213629607385 0.240408539138292 0.00681997559022603 -0.0027408943961907 -0.001619026613787 1149.14071676148 1148.7989685676 519.815837182153 501.402658888552 -0.208338188251856 0.255488007488945 -0.00246049749891915 -0.000759999321030303 0.00148438698385668 1145.51133842318 1144.77392807652 514.968197319863 501.882018537695 -0.198384093827848 0.218323676298049 -0.00894780704152122 -0.00181336200488089 -0.000587205583421232'
    x = np.array([float(x) for x in x.split(' ')])

    if type(subject) == str:
        subject = int(subject[1:])
    if type(camera) == str:
        camera_names = ['54138969', '55011271', '58860488', '60457274']
        camera = camera_names.index(camera) + 1

    # get the vector of coefficients for particular camera and subject
    w = np.zeros(15,)
    start = 6 * ((camera - 1) * 11 + subject-1)
    w[:6] = x[start:start+6]
    w[6:] = x[264+(camera-1)*9:264+camera*9]
    return w


def convert_camera(w):
    # Convert vector to more useful format. Code is adopted from
    # H36MDatabase.m and https://github.com/karfly/human36m-camera-parameters/blob/master/generate.py

    import cv2
    from scipy.spatial.transform import Rotation as R

    cam = {}
    cam['R'] = R.from_euler('XYZ', w[:3]).as_matrix()
    cam['T'] = -cam['R'] @ w[3:6]
    cam['f'] = w[6:8]
    cam['c'] = w[8:10]
    cam['k'] = w[10:13]
    cam['p'] = w[13:15]

    f = cam['f']
    c = cam['c']
    k = cam['k']
    p = cam['p']

    # make some useful mappings
    cam['K'] = np.array([[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1.0]])
    cam['dist'] = np.array([k[0], k[1], p[0], p[1], k[2]]) # order used by cv2
    cam['RT'] = np.concatenate([cam['R'], cam['T'].reshape(3,1)], axis=-1)
    cam['P'] = cam['K'] @ cam['RT']

    return cam


def import_data(data_dir):
    '''
    Import Human 3.6 into datajoint

    Params:
        data_dir (String) : path to extracted data

    This script expects to find the data already downloaded and extracted, which
    can be done with the scripts from https://github.com/CHUNYUWANG/H36M-Toolbox
    The path passed should be organized like:

    - Subject
      - Poses_D2_Positions
      - Poses_D3_Positions
      - Poses_D3_Positions_mono
      - Poses_D3_Positions_mono_universal
      - Videos

    Depends on spacepy, which is often best installed into a local directory and
    then run:

        os.environ['CDF_LIB'] = '/path/to/CDF/cdf38_0-dist/src/lib'
    '''

    import os
    from tqdm import tqdm
    from spacepy import pycdf

    videos = []
    subjects = os.listdir(data_dir)
    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject, 'Videos')

        def parse(f):
            activity, cam, _ = f.split('.')
            s = activity.split(' ')
            if len(s) == 1:
                trial = 0
            else:
                activity, trial = s
            return {'filename': os.path.join(subject_dir, f), 'store_filename': f'{subject}_{f}',
                    'subject_id': int(subject[1:]), 'activity': activity, 'trial': trial, 'cam': cam}

        videos.extend([parse(f) for f in os.listdir(subject_dir) if 'ALL' not in f])


    def fetch_3d(key):
        subject, remainder = key['store_filename'].split('_')
        activity, cam, _ = remainder.split('.')

        cdf_file = os.path.join(data_dir, f'{subject}/Poses_D3_Positions/{activity}.cdf')
        with pycdf.CDF(cdf_file) as cdf:
            poses_3d = np.array(cdf['Pose'])
            poses_3d = poses_3d.reshape(poses_3d.shape[1], 32, 3)

        return poses_3d # , cdf_file


    def fetch_2d(key):
        subject, remainder = key['store_filename'].split('_')
        activity, cam, _ = remainder.split('.')

        cdf_file = os.path.join(data_dir, f'{subject}/Poses_D2_Positions/{activity}.{cam}.cdf')
        with pycdf.CDF(cdf_file) as cdf:
            poses_2d = np.array(cdf['Pose'])
            poses_2d = poses_2d.reshape(poses_2d.shape[1], 32, 2)

        return poses_2d


    for k in tqdm(videos):

        if len(Subject & k) == 0:
            Subject.insert1({'subject_id': k['subject_id']})

        if len(Activity & k) == 0:
            keypoints3d = fetch_3d(k)
            Activity.insert1({'subject_id': k['subject_id'], 'activity': k['activity'],
                              'trial': k['trial'], 'keypoints3d': keypoints3d})

        if len(ActivityView & k) == 0:

            vid_struct = {'video_project': 'h36m', 'filename': k['store_filename'],
                          'start_time': "2021-07-25", 'video': k['filename']}

            if len(Video & vid_struct) == 0:
                Video.insert1(vid_struct, skip_duplicates=True)

            keypoints2d = fetch_2d(k)
            w = get_camera(k['subject_id'], k['cam'])
            calibration = convert_camera(w)
            ActivityView.insert1({'subject_id': k['subject_id'], 'activity': k['activity'],
                                  'trial': k['trial'], 'camera': k['cam'], 'keypoints2d': keypoints2d,
                                  'video_project': 'h36m', 'filename': k['store_filename'],
                                  'camera_calibration': calibration})
