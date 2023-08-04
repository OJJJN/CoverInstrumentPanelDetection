import cv2
import numpy as np
import argparse
import time

model = 'best.onnx'
img_w = 640
img_h = 640
classes_file = 'classes.txt'


def class_names():
    classes = []
    with open(classes_file, 'r') as file:
        for line in file:
            name = line.strip('\n')
            classes.append(name)
    return classes


width_frame = 640
net = cv2.dnn.readNetFromONNX(model)
classes = class_names()

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()

    height = int(frame.shape[0] * (width_frame / frame.shape[1]))
    dim = (width_frame, height)
    img = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    blob = cv2.dnn.blobFromImage(img, 1/255, (img_w, img_h), swapRB=True, mean=(0, 0, 0), crop=False)
    net.setInput(blob)
    t1 = time.time()
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    t2 = time.time()
    out = outputs[0]
    n_detections = out.shape[1]
    height, width = img.shape[:2]
    x_scale = width / img_w
    y_scale = height / img_h
    conf_threshold = 0.7
    score_threshold = 0.5
    nms_threshold = 0.5

    class_ids = []
    score = []
    boxes = []

    for i in range(n_detections):
        detect = out[0][i]
        confidence = detect[4]
        if confidence >= conf_threshold:
            class_score = detect[5:]
            class_id = np.argmax(class_score)
            if class_score[class_id] > score_threshold:
                score.append(confidence)
                class_ids.append(class_id)
                x, y, w, h = detect[0], detect[1], detect[2], detect[3]
                left = int((x - w/2) * x_scale)
                top = int((y - h/2) * y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, np.array(score), conf_threshold, nms_threshold)

    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        left, top, width, height = boxes[i]

        cv2.rectangle(img, (left, top), (left + width, top + height), (0, 0, 255), 2)
        label = "{}".format(classes[class_ids[i]])
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        dim, baseline = text_size[0], text_size[1]
        cv2.rectangle(img, (left, top - 20), (left + dim[0], top + dim[1] + baseline - 20), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, label, (left, top + dim[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('image_front_take', img)
    cv2.namedWindow("image_front_take")
    cv2.moveWindow("image_front_take", 0, 270)

    keyVal = cv2.waitKey(1) & 0xFF
    if keyVal == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()