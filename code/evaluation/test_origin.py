from collections import namedtuple
import mxnet as mx
import cv2
import numpy as np
import os

batch_size=1
ctx=mx.cpu()

json_path='/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/code/model_rgb/model_20-symbol.json'
params_path='/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/code/model_rgb/model_20-0000.params'

symnet=mx.symbol.load(json_path)
mod = mx.mod.Module(symbol=symnet,context=ctx)
mod.bind(data_shapes=[('data',(batch_size,3,64,64))])
mod.load_params(params_path)
Batch=namedtuple('Batch',['data'])

# test images path
img_path='/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/tools/datasets_test/rgbinfra_downstairs/image/5399/'

for img_name in os.listdir(img_path):
    path=img_path + img_name
    #img = cv2.imread(path,1)
    img = cv2.imread(path,1).transpose((2,0,1))
    img = cv2.resize(img,(64,64))
    img = img[np.newaxis]
    img = img[np.newaxis]
    
    # as you do normalization during training porcess, do it in testing
    
    mod.forward(Batch([mx.nd.array(img).reshape((1,3,64,64))]),is_train=False)

    pred = mod.get_outputs()
    pred_landmark = pred[0].asnumpy()[0]
    
    pred_landmark_point = np.array(pred_landmark)
    pred_landmark_point = 64 * pred_landmark_point
    pred_landmark_point = pred_landmark_point.reshape((15,2))
        
    img_plot=cv2.imread(path,1)
    img_plot = cv2.resize(img_plot,(64,64))
    for i in range(15):
        cv2.circle(img_plot,(int(pred_landmark_point[int(i)][0]),int(pred_landmark_point[int(i)][1])),1,(0,0,255),-1)
    if not os.path.exists('/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/code/results/'):
        os.mkdir('/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/code/results/')
    cv2.imwrite('/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/code/results/'+img_name,cv2.resize(img_plot,(256,256)))






