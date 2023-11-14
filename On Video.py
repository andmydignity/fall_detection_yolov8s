from ultralytics import YOLO
import cv2
#Link=https://sghub.deci.ai/models/yolo_nas_pose_s_coco_pose.pth

# Load a model
model = YOLO('yolov8s-pose.pt',task="pose")  # pretrained YOLOv8n model

# Open the video file
video_path = ''  # Replace with the path to your video file
results = model(video_path, stream=True,save=True,device="cuda")
frame=1
fall=0
for result in results:
    img = result.orig_img
    try:
        boxes = result.boxes  # Boxes object for bbox outputs
        for box in boxes:
            x = boxes.xywhn[0][0]
            y = boxes.xywhn[0][1]
            w = boxes.xywhn[0][2]
            h = boxes.xywhn[0][3]
            if w/h > 1.4:
                fall+=1
                print("Fall detected at {} frame".format(frame))
                
                #Print fall on top of persons head
                cv2.putText(img, "Fall", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                
            else:
                cv2.putText(img, "Stable", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #cv2.imwrite("frames/frame_{}.jpg".format(frame), img)
    except:
        pass
        #cv2.imwrite("frames/frame_{}.jpg".format(frame), img)

    frame+=1

print("Total fall detected: {}".format(fall))