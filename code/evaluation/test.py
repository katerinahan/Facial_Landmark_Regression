import os
import cv2
import mxnet as mx
import collections
import numpy as np
import sys
import dlib

batch_size = 1
ctx = mx.cpu()

json_path = 'C:/Users/yutong.han/Desktop/model_180-symbol.json'
params_path = 'C:/Users/yutong.han/Desktop/model_180-0000.params'

symnet = mx.symbol.load(json_path)
mod = mx.mod.Module(symbol=symnet,context=ctx)
mod.bind(data_shapes=[('data',(batch_size,3,64,64))])
mod.load_params(params_path)
Batch = collections.namedtuple('Batch',['data'])

img_path = 'C:/Users/yutong.han/Desktop/5392/'
cnn_face_detector = dlib.get_frontal_face_detector()

for img_name in os.listdir(img_path):
    path = img_path + img_name
    list = [path]
    for f in list:
        img = cv2.imread(f)
        b, g, r = cv2.split(img)
        img2 = cv2.merge([r, g, b])

        dets = cnn_face_detector(img, 1)
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

            #cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)

            cv2.imwrite('C:/Users/yutong.han/Desktop/detect_results/' + img_name, img[top:bottom,left:right])

            crop_face = cv2.imread('C:/Users/yutong.han/Desktop/detect_results/' + img_name,1)
            crop_face = cv2.resize(crop_face,(64,64))
            crop_face = crop_face.transpose((2, 0, 1))
            crop_face = crop_face[np.newaxis]

            mod.forward(Batch([mx.nd.array(crop_face).reshape((batch_size, 3, 64, 64))]), is_train=False)
            pred = mod.get_outputs()

            pred_landmark_scale = pred[0].asnumpy()[0]
            pred_landmark_point_scale = np.array(pred_landmark_scale)
            pred_landmark_point_scale = pred_landmark_point_scale.reshape((15, 2))

            if not os.path.exists('C:/Users/yutong.han/Desktop/ldmk/'):
                os.mkdir('C:/Users/yutong.han/Desktop/ldmk/')
            img_plot = cv2.imread(f,1)
            for j in range(15):
                cv2.circle(img_plot, (int(pred_landmark_point_scale[int(j)][0]*(right-left)+left), int(pred_landmark_point_scale[int(j)][1]*(bottom-top)+top)), 4,
                           (0, 0, 255), -1)
            cv2.rectangle(img_plot, (left, top), (right, bottom), (0, 255, 0), 4)
            cv2.imwrite('C:/Users/yutong.han/Desktop/ldmk/' + str(i) + '_' + img_name, img_plot)



