import cv2
import numpy as np
import serial

# Serial connection to ESP32
ser = serial.Serial('/dev/ttyUSB0',115200)
#ser = None
# Load class names
with open("model/coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO model
net = cv2.dnn.readNet("model/yolov3-tiny.weights","model/yolov3-tiny.cfg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame,1/255,(416,416),(0,0,0),True,crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.6:

                label = classes[class_id]

                if label in ["bird","cow","dog","sheep","horse","person"]:

                    print("Detected:",label)

                    if label == "person":
                        msg = "HUMAN\n"
                    elif label == "bird":
                        msg = "BIRD\n"
                    else:
                        msg = "ANIMAL:"+label.upper()+"\n"

                    ser.write(msg.encode())

    cv2.imshow("AI Detection",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()
