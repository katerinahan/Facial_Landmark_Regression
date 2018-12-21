# -*- coding:utf-8 -*-

import os
import cv2
import mxnet as mx
import collections
import numpy as np
import sys
import dlib
import time

batch_size = 1
ctx = mx.cpu()

json_path = 'C:/Users/yutong.han/Desktop/katerina/model/model_240-symbol.json'
params_path = 'C:/Users/yutong.han/Desktop/katerina/model/model_240-0000.params'

symnet = mx.symbol.load(json_path)
mod = mx.mod.Module(symbol=symnet,context=ctx)
mod.bind(data_shapes=[('data',(batch_size,3,64,64))])
mod.load_params(params_path)
Batch = collections.namedtuple('Batch',['data'])

classifier = cv2.CascadeClassifier('C:/Users/yutong.han/AppData/Local/Continuum/anaconda3/envs/workspace'
                                   '/Library/etc/haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    if not cap.isOpened():
        pass

    ret, frame = cap.read()

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0

    while ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        color = (0, 255, 0)

        faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        #print(faceRects)
        if len(faceRects):
            for faceRect in faceRects:
                x, y, w, h = faceRect

                #crop_face = frame[x:x+ h, y:y+ w] # False Crop
                crop_face = frame[y:y + h,x:x + w]

                crop_face = cv2.resize(crop_face, (64, 64))
                crop_face = crop_face.transpose((2, 0, 1))
                crop_face = crop_face[np.newaxis]

                mod.forward(Batch([mx.nd.array(crop_face).reshape((batch_size, 3, 64, 64))]), is_train=False)
                pred = mod.get_outputs()

                pred_landmark_scale = pred[0].asnumpy()[0]
                pred_landmark_point_scale = np.array(pred_landmark_scale)
                pred_landmark_point_scale = pred_landmark_point_scale.reshape((15, 2))

                for j in range(15):
                    cv2.circle(frame, (int(pred_landmark_point_scale[int(j)][0] * w + x),
                                            int(pred_landmark_point_scale[int(j)][1] * h + y)), 4,
                                (0, 0, 255), -1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.imshow("cap", frame)
        cv2.waitKey(int(1000 / int(fps)))  # 延迟
        ret, frame = cap.read()  # 获取下一帧
