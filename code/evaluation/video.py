import os
import cv2
import mxnet as mx
import collections
import numpy as np
import sys
import dlib

batch_size = 1
ctx = mx.cpu()

json_path = 'C:/Users/yutong.han/Desktop/model_240-symbol.json'
params_path = 'C:/Users/yutong.han/Desktop/model_240-0000.params'

symnet = mx.symbol.load(json_path)
mod = mx.mod.Module(symbol=symnet,context=ctx)
mod.bind(data_shapes=[('data',(batch_size,3,64,64))])
mod.load_params(params_path)
Batch = collections.namedtuple('Batch',['data'])

cnn_face_detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)

frame_index = 0
while 1:
    ret, frame = cap.read()
    if ret is False:
        continue
    dets = cnn_face_detector(frame, 1)
    print("Number of faces detected: {}".format(len(dets)))

    for i, face in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, face.left(), face.top(), face.right(), face.bottom()))
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()

        cx = (right + left) / 2
        cy = (bottom + top) / 2

        scale = 1.156789
        max_side = max(right - left, bottom - top)
        max_side = max_side * scale
        left = int(cx - max_side / 2)
        top = int(cy - max_side / 2)
        right = int(cx + max_side / 2)
        bottom = int(cy + max_side / 2)

        #cv2.imwrite('C:/Users/yutong.han/Desktop/detect_results/' + str(frame_index) + '.jpg', frame[top:bottom, left:right])
        #crop_face = cv2.imread('C:/Users/yutong.han/Desktop/detect_results/' + str(frame_index) + '.jpg', 1)
        crop_face = frame[top:bottom, left:right]
        crop_face = cv2.resize(crop_face, (64, 64))
        crop_face = crop_face.transpose((2, 0, 1))
        crop_face = crop_face[np.newaxis]

        mod.forward(Batch([mx.nd.array(crop_face).reshape((batch_size, 3, 64, 64))]), is_train=False)
        pred = mod.get_outputs()

        pred_landmark_scale = pred[0].asnumpy()[0]
        pred_landmark_point_scale = np.array(pred_landmark_scale)
        pred_landmark_point_scale = pred_landmark_point_scale.reshape((15, 2))

        for j in range(15):
            cv2.circle(frame, (int(pred_landmark_point_scale[int(j)][0] * (right - left) + left),
                                  int(pred_landmark_point_scale[int(j)][1] * (bottom - top) + top)), 4,
                       (0, 0, 255), -1)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 4)
        cv2.imshow("cap", frame)
        if cv2.waitKey(100) & 0xff == ord('q'):
            break
#cap.release()
#cv2.destroyAllWindows()
