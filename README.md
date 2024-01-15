# Object Tracking

Track moving object in a given video.

## Requirements
* Numpy >= 1.24.3
* OpenCV >= 4.9.0 (latest as of 2024)

## Description
We have implemented Object Tracking Software which tracks moving object
in a given video. We have implemented in two ways:
1. From scratch using **Lucas Kanade Tracking** algorithm which uses six parameter affine model
   and recursive Gauss-Newton algorithm. Implemented the research paper Lucas-Kanade 20 Years On: by 
   simon Baker (Microsoft Computer vision researcher).
2. using OpenCV library.
 
Then, We have done the detailed comparative analysis of both implementation which can be found in the attached pdf.

## How To Run
1. Put video in `video/` folder, or you can use provided sample video `slow_traffic.mp4` .
2. Run:
   ```shell
   python3 lucas_kanade_scratch.py
   ```
   or
   ```shell
   python3 lucas_kanade_opencv.py
   ```
3. Enter video filename along with path like:
   ```shell
   Enter video filename along with path: video/slow_traffic.mp4
   ```
   Enter "0" to use your real time system camera for video.