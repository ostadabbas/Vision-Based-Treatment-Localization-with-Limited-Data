<div align="center">
<img width=30% src="media/SOCOM Ignite_logo_2023.png">

<br> </br>

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/release/python-3100/) ![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/socom-ignite/csct-cv/build.yml?branch=v5testbed&style=for-the-badge)
</div>

## Basic Overview
The computer vision models for the CSCT project are essential in the identification of external medical interventions. Using live camera feeds, we process the frames through various models to both identify and localize the site of the interventions. To achieve this, we will use already well-developed models such as YOLO, and OpenPose to detect, and pose estimation respectively. Being a hands free method of filling out multiple sections of the TCCC Card, this will save time and people from needing to help fill out cards as well having digital copies during patient handoff better ensuring correct and important information gets passed on.

## Branch Overview
This branch is our code submission for ICCV. It is the merge between object detection using yolov5 and pose estiamtion using mobilenet. We feed the coordinates of yolov5 detections into mobilenet to localize the detection to a segment on the body.

# Getting Started (REQUIRED: Python >= 3.10)

## Clone Git Repository
```
git clone https://github.com/SOCOM-Ignite/CSCT-CV.git
git checkout -b yolo-pose-merge
cd CSCT-CV/pipeline
pip install python -r requirements.txt
```

## Running demo for testing purposes
For a basic run using the base human pose estiamtion input the following

```
python demo.py --hpe
```

Please run `python demo.py --help` for a iist of arguments and descriptors.

## Running live demo
For running the live demo, please ensure that you have a webcam and microphone readily availible as well as port 8080 not in use. 
Run the following:
```
python live_demo.py
```

## Webserver
When running navigate to [http://localhost:8080/mjpg](http://localhost:8080/mjpg) to view the mjpg stream and the opencv outputs overlayed. When the pipeline is started the img will switch to the feed, when it is turned off it will switch back to the no signal image.

To view the json file navigate to [http://localhost:8080/json](http://localhost:8080/json). If available the json will be posted, if not the pipeline is still running. The json file will be created/updated twice after the pipeline ends: once after cv processes treatments, and once after whisper process drugs/blood work

Please run `python live_demo.py --help` for a iist of arguments and descriptors.


## Pose Estimation
The image below is the skeleton generation on the full body and the reference keypoint values. The following is based upon the COCO keypoints-18 labeling system

<div align="center">
<img align="center" width=30% src="media/COCOKeypoints18.png">
</div>

We use a lightweight CPU optimized version of CMU's OpenPose software as the base to our human pose estimation algorithms. The model has been trained on the COCO keypoints dataset, and finetuned on ~1000 custom labeled images across our various datasets. In our use case, the camera is located within 1 meter of the subject therefore often leading to partial poses or less. We  compensate the lack of keypoints detected using various [augmentations](https://github.com/SOCOM-Ignite/CSCT-CV/tree/v5testbed/pipeline/utils/pose/augmentations).

## Organization
All code is in `CSCT-CV/pipeline` directory. Run outputs are generated in the `outputs/` folder and
temporary files are created and destroyed in the `tmp/` folder. 


Code is seperate by component in the following pipeline (shown below) to better conform to a less messy OOP. For example, the bulk of the HPE code will be located in the HPE class in hpe.py in the `/pipeline/components` folder.
<p align="center"><img width=100% src="media/Pipeline Summary 29MAR23.png"></p>

## Contributing
Feel free to contribute any working improvements (does not have to be PR just dont break it) and please use commit messages

#### Pending Features
- Localization when more than 1 skeleton is generated
- Localization when bounding box is on a segment that is not identified by pose estimation
- Partial Pose Estimation 
- Better trained weights

## Citations:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7347926.svg)](https://doi.org/10.5281/zenodo.7347926) 
[![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.1811.12004.svg)](https://doi.org/10.48550/arXiv.1811.12004)