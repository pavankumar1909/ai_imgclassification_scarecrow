import cv2
import numpy as np
import serial

ser = serial.Serial('/dev/ttyUSB0',115200)

with open("model/coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

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

    boxes = []
    confidences = []
    class_ids = []

    # Collect detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.6:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Remove duplicate overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = classes[class_ids[i]]

            if label in ["bird","cow","dog","sheep","horse","person"]:

                # Draw rectangle
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                # Put label text
                cv2.putText(frame,f"{label} {int(confidences[i]*100)}%",
                            (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

                print("Detected:",label)

                if label == "person":
                    msg = "HUMAN\n"
                elif label == "bird":
                    msg = "BIRD\n"
                else:
                    msg = "ANIMAL:"+label.upper()+"\n"

                if ser is not None:
                    ser.write(msg.encode())

    cv2.imshow("AI Detection",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()
