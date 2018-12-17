# -*- coding: utf-8 -*-

'''
All of codes below are written by Haofan Wang.
Copyright reserved.
'''

import json
import numpy as np
import cv2
import os
from PIL import ImageFile
import random
import argparse
import utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Predicting.')
    parser.add_argument('--size', dest='size', help='size of image',
                        default=64, type=int)
    parser.add_argument('--lst', dest='lst', help='lst location ',
                        default='/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/code/train.lst', type=str)
    parser.add_argument('--json', dest='json', help='json location ',
                        default= '/mnt/hdfs-data-3/data/zhenghua.chen/DMS_Data/Data_All/', type=str)
    parser.add_argument('--image', dest='image', help='image location ',
                        default='/mnt/hdfs-data-3/data/zhenghua.chen/DMS_Data/Data_All/', type=str)
    parser.add_argument('--save', dest='save', help='image save location ',
                        default='/mnt/hdfs-data-1/adas/haofan.wang/facial_landmark/tools/datasets/', type=str)
    args = parser.parse_args()
    return args


args = parse_args()
size = args.size
lst_location = args.lst
json_root =args.json
image_root = args.image
save_path = args.save

keypoint_15 = [0,2,3,5,6,8,10,11,13,15,16,19,20,21,22]

'''
生成image-json字典：
        json_image = {'json_path':'image_path'}
'''
data_folder_names = []
json_abs_location = []
image_folder_location = []
json_image = {}

for root,dirs,files in os.walk(json_root):
    data_folder_names = dirs
    break

# 去掉image/json内容缺失的文件夹
data_folder_names.remove('ruilian_njoutside_20180926')

# 作为test集
data_folder_names.remove('lightside_njsmoke_20181102')

# 处理版本不统一问题
for i in range(len(data_folder_names)):
    if 'v20181129' in os.listdir(json_root+data_folder_names[i]+'/json'):
      for json_file in os.listdir(json_root+data_folder_names[i]+'/json'+'/'+'v20181129'):
        json_abs_location.append(json_root+data_folder_names[i]+'/json'+'/'+'v20181129'+'/'+json_file)
        image_folder_location.append(image_root + data_folder_names[i] + '/image/' + json_file[0:4])
    elif 'v20181128' in os.listdir(json_root+data_folder_names[i]+'/json'):
      for json_file in os.listdir(json_root+data_folder_names[i]+'/json'+'/'+'v20181128'):
        json_abs_location.append(json_root+data_folder_names[i]+'/json'+'/'+'v20181128'+'/'+json_file)
        image_folder_location.append(image_root + data_folder_names[i] + '/image/' + json_file[0:4])
    elif 'v20181127' in os.listdir(json_root+data_folder_names[i]+'/json'):
        for json_file in os.listdir(json_root+data_folder_names[i]+'/json'+'/'+'v20181127'):
          json_abs_location.append(json_root+data_folder_names[i]+'/json'+'/'+'v20181127'+'/'+json_file)
          image_folder_location.append(image_root + data_folder_names[i] + '/image/' + json_file[0:4])

# json -- image 字典
for json_file in json_abs_location:
    (filepath,tempfilename) = os.path.split(json_file)
    (filename,extension) = os.path.splitext(tempfilename)
    json_image[json_file] = filepath[:len(filepath)-14]+'image/'+filename + '/'

with open('data.lst', 'w+') as lst:
  count = 0
  
  # 存储每一张图片对应的label数据，写入lst
  label = []
  
  # 遍历所有json文件
  for json_path in json_image.keys():
      
      # 存储该json文件中所有有效数据
      tmp_list = []
      
      # 剔除该json文件中脏数据
      with open(json_path, 'r') as f:
          for line in f:
              if line[0] == '#':
                  continue
              else:
                  tmp_list.append(json.loads(line))
      
      # 该json文件有效长度
      length = len(tmp_list)
      
      landmarks_dic = {}
      
      # 遍历该json文件中每一行，存储landmark数据
      for i in range(length):
          landmarks_dic[tmp_list[i]['image_key']] = []
          ignore_time = 0
          
          # 'head'不存在时，剔除
          if 'head' not in tmp_list[i].keys():
              continue
              
          # 'ignore'为no的个数不为1时，剔除
          for j in range(len(tmp_list[i]['head'])):
              if tmp_list[i]['head'][j]['attrs']['ignore'] == 'no':
                  ignore_time += 1
                  
          # 当一张图片中存在多个人时，遍历，选择ignore为no的司机图像
          for j in range(len(tmp_list[i]['head'])):
              if tmp_list[i]['head'][j]['attrs']['ignore'] == 'no' and ignore_time==1 and 'face_keypoint_28' in tmp_list[i].keys():
                  landmarks_dic[tmp_list[i]['image_key']].append(tmp_list[i]['face_keypoint_28'][0]['data'])
              else:
                  continue

      # 保存crop的人脸图像
      image_path = json_image[json_path]
      root_len = len('/mnt/hdfs-data-3/data/zhenghua.chen/DMS_Data/Data_All/')
      new_sample_path = save_path + image_path[root_len:]
      
      # 如果路径不存在，则创建
      if not os.path.exists(new_sample_path):
          os.makedirs(new_sample_path)
      files = os.listdir(image_path)
      
      # crop的尺寸设定
      img_width = size
      img_height = size
      
      for image in files:
          full_path = image_path + image
          
          # 避免读取图片时可能出现的"image file is truncated"错误
          ImageFile.LOAD_TRUNCATED_IMAGES = True
          
          img = cv2.imread(full_path,1)
          key = image
          #key = image[:len(image) - 3] + 'png'
          
          if key not in landmarks_dic.keys():
            continue
          
          if len(landmarks_dic[key])==0 or len(landmarks_dic[key][0])!=28:
            continue
          
          if type(img) != type(None):
            center = landmarks_dic[key][0][16]
            lmks = np.array(landmarks_dic[key][0]).reshape((28,2))
            box = []
            for j in range(lmks.shape[0]):
              if not box:
                box = [lmks[j,0],lmks[j,1],lmks[j,0],lmks[j,1]]
              else:
                box[0] = min(box[0],lmks[j,0])
                box[1] = min(box[1],lmks[j,1])
                box[2] = max(box[2],lmks[j,0])
                box[3] = max(box[3],lmks[j,1])
                  
            if center is not None:
              center_x = center[0]
              center_y = center[1]
            else:
              center_x = (box[0]+box[2])/2
              center_y = (box[1]+box[3])/2
            
            tmp_box = box.copy()
            tmp1_box = box.copy()
            scale = 1.25
            half_len = scale*(tmp_box[3]-tmp_box[1]+tmp_box[2]-tmp_box[0])/4
            tmp_box[0] = int(center_x-half_len)
            tmp_box[1] = int(center_y-half_len)
            tmp_box[2] = int(center_x+half_len)
            tmp_box[3] = int(center_y+half_len)
            
            max_slide = half_len*2
            x_max = tmp_box[2]
            x_min = tmp_box[0]
            y_max = tmp_box[3]
            y_min = tmp_box[1]

            # 剔除数据集中少量脏数据
            if max_slide == 0:
              continue
            if x_min < 0 or y_min < 0:
              scale = 1.10
              half_len = scale*(tmp1_box[3]-tmp1_box[1]+tmp1_box[2]-tmp1_box[0])/4
              tmp1_box[0] = int(center_x-half_len)
              tmp1_box[1] = int(center_y-half_len)
              tmp1_box[2] = int(center_x+half_len)
              tmp1_box[3] = int(center_y+half_len)
              
              max_slide = half_len*2
              x_max = tmp1_box[2]
              x_min = tmp1_box[0]
              y_max = tmp1_box[3]
              y_min = tmp1_box[1]
            if max_slide == 0:
              continue
            if x_min < 0 or y_min < 0:
              continue
              
            crop_img = img[y_min:y_max, x_min:x_max]
            crop_img = cv2.resize(crop_img,(img_height,img_width))
            cv2.imwrite(new_sample_path + image,crop_img)
            
            # 打印label到crop图片上，验证label的准确性
            '''
            plot_img = crop_img.copy()
            for index in keypoint_15:
              cv2.circle(plot_img,
                         (
                          int(64*((lmks[int(index)][0]-x_min)/float(x_max-x_min))),
                          int(64*((lmks[int(index)][1]-y_min)/float(y_max-y_min)))
                          ),
                          2,(0,0,255),-1)
            cv2.imwrite(new_sample_path + image,plot_img)
            '''
            
            write_line = ''
            write_line += str(count)
            write_line += '\t'
            
            # 使用(x_min,y_min)作为基准点，landmark归一化到(0,1)
            for index in keypoint_15:
              write_line += str((lmks[int(index)][0]-x_min)/float(x_max-x_min))
              write_line += '\t'
              write_line += str((lmks[int(index)][1]-y_min)/float(y_max-y_min))
              write_line += '\t'
            
            # 使用(cx,cy)作为基准点，landmark归一化到(-0.5,0.5)
            '''
            for index in keypoint_15:
              write_line += str((lmks[int(index)][0]-center_x)/float(x_max-x_min))
              write_line += '\t'
              write_line += str((lmks[int(index)][1]-center_y)/float(y_max-y_min))
              write_line += '\t'
            '''
            
            lst.write(write_line + new_sample_path + image + '\n')
            count += 1
            
            if count%200 == 0:
              print(count)