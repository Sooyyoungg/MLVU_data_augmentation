#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import glob
import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree
import pandas as pd


# In[18]:


my_dict= {}
my_dict['aeroplane'] = 1
my_dict['bicycle'] = 2
my_dict['bird'] =3
my_dict['boat'] = 4
my_dict['bottle']= 5
my_dict['bus'] = 6
my_dict['car'] = 7
my_dict['cat'] = 8
my_dict['chair'] = 9
my_dict['cow'] = 10
my_dict['diningtable'] =11
my_dict['dog'] =12
my_dict['horse'] = 13
my_dict['motorbike'] = 14
my_dict['person'] = 15
my_dict['pottedplant'] =16
my_dict['sheep'] = 17
my_dict['sofa'] = 18
my_dict['train'] = 19
my_dict['tvmonitor'] = 20


# In[19]:


xml_path = '/share/home/connectome/conmaster/MLVU/mmdetection/data/VOCdevkit/VOC2012/Annotations/*.xml'
file_list = glob.glob(xml_path)
print(len(file_list))
print(file_list)


# In[20]:


print("XML parsing Start\n")

py_dict = {}
for file_name in file_list:
    xml = open(file_name, "r")
    tree = Et.parse(xml)
    root = tree.getroot()

    segmented = root.find("segmented").text
    
    # if the image was segmented
    if segmented =='1':
        print(file_name)

        # find all of the objects in a xml file
        print("<Objects Description>")
        objects = root.findall("object")
        name_list= []
        for _object in objects:
            name = _object.find("name").text
            name_list.append(name)
        
        print("class : ", name_list)
        print("\n")    
        py_dict[file_name] = name_list


# In[21]:


empty = []
f = open("/share/home/connectome/conmaster/MLVU/Data/train_file.txt", 'r')
lines = f.readlines()
for line in lines:
    empty.append(line[:-1])
f.close()
print(len(empty))
print(empty)


# In[22]:


img_path = '/share/home/connectome/conmaster/MLVU/mmdetection/data/VOCdevkit/VOC2012/JPEGImages/'
seg_path = '/share/home/connectome/conmaster/MLVU/mmdetection/data/VOCdevkit/VOC2012/SegmentationClass/' 


# In[23]:


stat_dist=np.zeros([20, 20], dtype=int)

distribution=pd.DataFrame(stat_dist, columns=my_dict.keys(), index=my_dict.keys())
print(distribution)


# In[24]:


# 이런 식으로 호출해서 값 추가할 예정
distribution['car']['sheep']


# In[25]:


"""count the number of appearence of each object when two or more objects """
count_iso=np.zeros([1,20], dtype=int)
count_iso=pd.DataFrame(count_iso, columns=my_dict.keys())

# for all images
for key in py_dict.keys():
    image_name = key[-15:-4]
    
    # if image is in dataset
    if str(image_name) in empty:
        objects=py_dict[key]
        
        if len(objects) > 1:
            # add value of two objects in one image to distribution matrix
            for obj in objects:
                for other_objs in objects:
                    # Isomorphic matrix
                    distribution[obj][other_objs] += 1
                distribution[obj][obj] -= 1


# In[26]:


distribution


# In[31]:


distribution.shape


# In[27]:


############ Training ############
"""make training data"""
## Image
# 1. Image -> numpy array
# 2. concatenate 3 channel original images and 1 channel segmentation annotation
# 3. resize to unify the sizes of input images
# 4. stack all the image information

## Class
# 1. make an array for each image : 1 x 20
# 2. count the number of detection for each objects
# 3. resize for stacking
# 4. normalize the arraies to make their sum is 1
# 5. stack all the class information

X=[]
Y=[]
count=0
for key in py_dict.keys():
    image_name = key[-15:-4]
    
    # if image is in dataset
    if str(image_name) in empty:
        
        """ Input Data """
        # get segment informations by pixel
        segmentation_path = seg_path + image_name + '.png'
        seg_image = Image.open(segmentation_path)
        array_pixel = np.array(seg_image)

        # get original images information by pixel
        image_path  = img_path + image_name + '.jpg'
        image = Image.open(image_path)
        image_pixel = np.array(image)
                
        # concatenate original image and segmentation annotation
        input_image = np.dstack((image_pixel, array_pixel))
        
        # resize the input images as the sizes of the images are different
        input_image=np.resize(input_image, (1, 500, 500, 4))
        
        # stack all the input images
        if count==0:
            X = input_image
        else:
            X = np.concatenate((X, input_image), axis=0)
        
        """ Label=Class data """
        input_class=np.zeros(20, dtype=float)
        for obj in py_dict[key]:
            input_class[my_dict[obj]-1]+=1
        input_class = np.resize(input_class, (1, 20))
        
        # normalize the values whose sum has to be 1
        T = np.copy(input_class)
        for i in range(len(my_dict.keys())):
            sums = input_class[0].sum()
            T[0][i] /= float(sums)
        
        # stack all the class informations
        if count==0:
            Y = T
            count+=1
        else:
            Y = np.concatenate((Y, T), axis=0)
        
X_train=np.array(X)
Y_train=np.array(Y)
print(X_train.shape)
print(Y_train.shape)


# In[39]:


distribution.to_csv("./distribution.csv", index=False)
np.save('./X_train', X_train)
np.save("./Y_train", Y_train)


# In[ ]:




