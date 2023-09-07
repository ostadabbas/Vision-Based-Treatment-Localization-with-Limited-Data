# Vision-Based Treatment Localization with Limited Data: Automated Documentation of Military Emergency Medical Procedures
*Authors: Trevor Powers, Elaheh Hatamimajoumerd, William Chu, Marc Vaillant,
Rishi Shah, Vishakk Rajendran, Frank Diabour, Richard Fletcher, Sarah Ostadabbas* 

<p align="center">
  <a href="#Introduction">Introduction</a> |
  <a href="#Framework">Framework</a> |
  <a href="#Main Results">Main Results</a> |
  <a href="#Experiment">Experiment</a> |
  <a href="#Acknowledgments">Acknowledgments</a> 
</p>

## Introduction
<p align="center">
<img src="figure/video.gif" >
</p>
This repository is the official repository of <a href=''>Vision-Based Treatment Localization with Limited Data: Automated Documentation of Military Emergency Medical Procedures</a>. 
In response to the challenges faced in documenting medical procedures in military settings, where time constraints and cognitive load limit the completion of life-saving Tactical Combat Casualty Care (TCCC) Cards, we present a novel end-to-end computer vision pipeline for autonomous detection and documentation of common military emergency medical treatments. Our pipeline is specifically designed to handle limited and challenging data encountered in military scenarios. To support the development of this pipeline, we introduce SimTrI, a labeled dataset comprising 116 twenty-second videos capturing patients undergoing four prevalent treatment procedures. Our pipeline incorporates training and fine-tuning of object detection and human pose estimation models, complemented by a proprietary pose-enhancement algorithm and a range of unique filtering and post-processing techniques. Through comprehensive development and optimization, our pipeline achieves exceptional performance, demonstrating 100\% precision and 62\% recall on our dedicated 23-video test set. Furthermore, the pipeline automates the generation of TCCC-relevant information, significantly improving the efficiency of TCCC documentation.

## Framework

![Alt Text](figure/full_pipeline_23Jun.jpg)
Illustration of our comprehensive pipeline for casualty status documentation. The pipeline consists of two main stages, shown from left to right. In the first stage (Pairing Matrix Creation), the input video is processed frame by frame, and relevant detections are analyzed and summarized to generate a pairing matrix. In the second stage (Video Post Processing), the summarized detections undergo post-processing to extract TCCC-relevant information. Subsequently, in the Results Generation stage, the pipeline generates a digital TCCC card formatted with the extracted information, and its metrics are reported based on the ground truth data.

## Main Results
| HPE Model | Z-Score Window Size | Min Distance RC | Min Pose Bbox Area RC | Min Number of Joints RC | c-threshold | PeTA Usage | Raw Precision | TCCC Precision | TCCC Recall | 
|--------|--------------|-------------|-----|------|------|-----------|--------|--------|------|
| Base | NA | 1       | 0    |97.3| 95.8 | 83.2 | 78.8      | 77.1   | 62.6   | 
| Base | 60 | 1      | 0    |97.5| 97.2 | 79.4 | 87.8      | 90.3   | 93.8  | 
| Base | 60 | 1      | 0    |**97.8**| **98.3** | 81.1 | 94.0      | 93.5   | 92.0   | 
| Base | 60 | .25       | .1 |**97.8**| 96.5 | **93.4** | **98.4** | **95.5** | 92.9   |
| Mann | 60 | .1       | .1 |93.1| 99.7 | 77.0 | 93.8      | 91.0   | 86.5   |
| Mann | 60 | .25      | .1 |**98.1**| **98.6** | 82.4 | **98.6**      | **97.8**   | **97.9**   |
| HuMann | 60 | .1      | .1 |99.6| 99.7 | **83.4** | 98.4      | 97.3   | 96.4   |
| HuMann | 60 | .25       | .3 |99.7| 98.0 | 81.8 | 97.5      | 96.4   | 97.1   |
| Mann | 60 | .5      | .1 |**98.1**| **98.6** | 82.4 | **98.6**      | **97.8**   | **97.9**   |



## License 
* This code is for non-commertial purpose only. 
* For further inquiry please contact: Augmented Cognition Lab at Northeastern University: http://www.northeastern.edu/ostadabbas/ 
