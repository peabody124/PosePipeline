

1. Install PosePipeline into your path

```
git clone https://github.com/peabody124/PosePipeline.git
cd PosePipeline
pip install -e .
```

2. Launch DataJoint database. One can also set up a local MySQL database, following the instruction at DataJoint.

```
cd datajoint_docker
docker-compose up -d
```

3. When running code, make sure to configure repository for where video stores will be kept. This can also be saved to the DataJoint configuration.

```
dj.config['stores'] = {
    'localattach': {
        'protocol': 'file',
        'location': '/mnt/data0/clinical_data/datajoint_external'
    }
}
```

4. Set the environment variables found in `pose_pipeline.env` based on the local installation. Follow the specific installations instructions for each one and also store the necessary network weights in the 3rdparty directory.

MMPose files:
```
3rdparty/mmpose/checkpoints/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth
3rdparty/mmpose/checkpoints/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth
3rdparty/mmpose/checkpoints/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth
3rdparty/mmpose/checkpoints/res50_coco_640x640-2046f9cb_20200822.pth
```
