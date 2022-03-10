# PosePipe: Open-Source Human Pose Estimation Pipeline for Clinical Research

![ERD](doc/erd.png)

PosePipe is a human pose estimation (HPE) pipeline to facilitate home movement analysis from videos. It uses [DataJoint](https://github.com/datajoint) to manage the interdependencies between algorithms and for data management of the videos and the intermediate outputs. If has wrappers to numerous cutting edge HPE algorithms and output visualizations, which makes it easy to analyze videos differently and determine the best algorithms to use.

## Getting Started

- Follow the [Installation instructions](INSTALL.md) to install PosePipe and launch a DataJoint MySQL database.
- Use the [Getting Started Notebook](doc/Getting%20Started.ipynb) to test the pipeline
