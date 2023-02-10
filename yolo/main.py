# importing cv2 library to perform object detection
import cv2 

# 
net=cv2.dnn.readNet('yolov4-3.2.0/yolov4.weights','yolov4-3.2.0/yolov4.cfg')

# Load classes 
model=cv2.dnn_DetectionModel(net)

# Set input parameters
model.setInputParams(size=(416,416), scale=1/255)


classes=[]
with open('yolov4-3.2.0/classes.txt','r') as file_object:
    for class_name in file_object.readlines():
        class_name=class_name.strip()
        classes.append(class_name)

print(classes)

cap=cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def click_button(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('position (', x, ',', y, ')')

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', click_button)

while True:

    
    ret, frame=cap.read()

    (class_ids, scores, boxes)=model.detect(frame)
    for class_id, score, box in zip(class_ids, scores, boxes):
        (x,y,w,h)=box
        class_name=classes[class_id]

        print (x,y,w,h)
        cv2.putText(frame, class_name, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        # print (class_id, scores, boxes)

    cv2.imshow('Frame',frame)
    cv2.waitKey(1) 
   
