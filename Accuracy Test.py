from ultralytics import YOLO
import cv2
import pandas as pd
import time

#Link=https://sghub.deci.ai/models/yolo_nas_pose_s_coco_pose.pth

# Load a model
model = YOLO('yolov8s-pose.pt',task="pose")  # pretrained YOLOv8n model
def get_class(column,filepath):

    # Read the CSV file
    df = pd.read_csv(filepath)

    # Get the class value from column 1
    return df.iloc[column, 1]

source = 'Datasets/832/'
csv_path='Datasets/832.csv'
results = model(source, stream=True,device="cuda")
true=0
false_positive=0
false_negative=0
wrong_detection=0
frame=1
threshold=1.4
total_time = 0
for result in results:
    start_time = time.time()
    status=""
    try:
        boxes = result.boxes  # Boxes object for bbox outputs
        for box in boxes:
            x = boxes.xywhn[0][0]
            y = boxes.xywhn[0][1]
            w = boxes.xywhn[0][2]
            h = boxes.xywhn[0][3]
            if w/h > threshold:
                status="Fall"
            else:
                if status!="Fall":
                    status="Stable"
                
    except:
        status="Not Detected"
        #cv2.imwrite("frames/frame_{}.jpg".format(frame), img)
    if status=="Stable":
        if get_class(frame-1,csv_path)==1 or get_class(frame-1,csv_path)==2 or get_class(frame-1,csv_path)==4 or get_class(frame-1,csv_path)==5:
            true+=1
        else:
            false_negative+=1
    elif status=="Fall":
        if get_class(frame-1,csv_path)==3:
            true+=1
        else:
            false_positive+=1
    elif status=="Not Detected":
        if get_class(frame-1,csv_path)==6:
            true+=1
        else:
            false_positive+=1
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = 1 / elapsed_time
    total_time += elapsed_time
    frame+=1


avg_fps = frame / total_time
print("Accuracy: ",true/(true+false_positive+false_negative+wrong_detection),"False Positive: ",false_positive,"False Negative: ",false_negative,"Wrong Detection: ",wrong_detection,"Average FPS: {}".format(avg_fps))
