# fall_detection_yolov8s
A simple script i wrote that uses yolov8s-pose to find humans in image,get their bounding boxes' width and height and finally if width/height ratio is over 1.4 estimate that it is a fall.
It has a %96.3 accuracy rate,and runs pretty fast with CUDA. (For Nvidia RTX 3050 Mobile:385 FPS)
Dataset link: https://falldataset.com/
(i know this is not that user friendly but i just wanted to put this here in case it helps someone)
