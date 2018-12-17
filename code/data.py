import json
import cv2
import mxnet as mx
import subprocess
import os
import numpy as np
import glob

# json path
jsons_list = glob.glob('/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/tools/*.json')

# keypoint used
keypoint_15 = [0,2,3,5,6,8,10,11,13,15,16,19,20,21,22]

# travel each json and its correspoding images
if not os.path.exists('/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/code/train.lst'):
    os.mknod('train.lst')
    
with open('train.lst', 'w') as f:
    count = 0
    # read json
    for json_ in jsons_list:

      images_path = json_[:len(json_)-5] + '/'
      json_file = open(json_, 'r').readlines()
      # read lines
      for line in json_file:
        json_dict = json.loads(line)
        # only crop driver's face
        
        if 'head' not in json_dict.keys():
          continue
        if 'face_keypoint_29' not in json_dict.keys():
          continue
        if 'image_key' not in json_dict.keys():
          continue
        
        if json_dict['head'][0]['attrs']['ignore'] == 'no':
            key_points = json_dict['face_keypoint_29'][0]['data']
            image_name = json_dict['image_key']
            
            # face_box = [left_down_x, left_down_y, right_up_x, right_up_y]
            face_box = json_dict['head'][0]['data']
            # the maximum slide of face box, crop a square box
            length = int(max(face_box[2]-face_box[0],face_box[3]-face_box[1]))

            x1 = int(face_box[0])
            y1 = int(face_box[1])
            x2 = int(face_box[2])
            y2 = int(face_box[3])

            center = [int((x2+x1)/2),int((y2+y1)/2)]
            
            # read gray image to reduce computing amount
            if os.path.exists(images_path + image_name) is False:
              continue
            img = cv2.imread(images_path + image_name, 1)
            if img is None:
              continue

            x1 = int(center[0] - length / 2)
            y1 = int(center[1] - length / 2)
            x2 =  int(center[0] + length / 2)
            y2 = int(center[1] + length / 2)
            
            if x1<0 or y1<0 or x2>1280 or y2>720:
              continue
            
            # note the image array format, Not img[x1:x2,y1:y2]
            crop_img = img[y1:y2,x1:x2,:]
            # resize the small scale
            if crop_img.shape == np.array([]).reshape((0,0,3)).shape:
              continue
            crop_img = cv2.resize(crop_img,(64,64))
            crop_path = '/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/tools/11081_1_crop/'
            if not os.path.exists(crop_path):
              os.mkdir(crop_path)
            cv2.imwrite(crop_path + image_name,crop_img)
            
            '''
            for i in keypoint_15:
              cv2.circle(img,(int(key_points[int(i)][0]),int(key_points[int(i)][1])),2,(0,0,255),-1)
            plot = img[y1:y2,x1:x2,:]
            cv2.imwrite('/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/tools/11081_1_plot/'+ image_name,plot)
            '''
            
            '''
            write_line = ''
            for index in keypoint_15:
                # rescale the coordinate from raw image(720*1280) to cropped image(64*64)
                write_line += str((key_points[int(index)][0]-center[0])/float(x2-x1))
                write_line += '\t'
                write_line += str((key_points[int(index)][1]-center[1])/float(y2-y1))
                write_line += '\t'

            f.write(str(count) + '\t' + write_line + crop_path + image_name + '\n')
            count += 1
            '''
            write_line = ''
            for index in keypoint_15:
                # rescale the coordinate from raw image(720*1280) to cropped image(64*64)
                write_line += str((key_points[int(index)][0]-x1)/float(x2-x1))
                write_line += '\t'
                write_line += str((key_points[int(index)][1]-y1)/float(y2-y1))
                write_line += '\t'

            f.write(str(count) + '\t' + write_line + crop_path + image_name + '\n')
            count += 1

im2rec_path=os.path.join(mx.__path__[0],'/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/tools/im2rec.py')
if not os.path.exists(im2rec_path):
    im2rec_path=os.path.join(os.path.dirname(os.path.dirname(mx.__path__[0])),'/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/tools/im2rec.py')
subprocess.check_call(['python',im2rec_path,os.path.abspath('train.lst'),os.path.abspath('./'),'--pack-label'])
