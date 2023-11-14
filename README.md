# fall_detection_yolov8s
Script that employs yolov8s-pose to detect humans in an image, retrieve the width and height of their bounding boxes, and ultimately determine if the width/height ratio exceeds 1.4, indicating a potential fall.
It has a %96.3 accuracy rate,and runs pretty fast with CUDA. (For Nvidia RTX 3050 Mobile:385 FPS)
Dataset link: https://falldataset.com/
(i know this is not that user friendly but i just wanted to put this here in case it helps someone)
