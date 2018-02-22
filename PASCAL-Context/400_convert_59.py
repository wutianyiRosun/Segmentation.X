#encoding:utf-8
#python 2.7
#author: Rosun
import sys
import os
import numpy as np
from PIL import Image
from scipy.io import loadmat as sio
import random

def PascalContext_400class_Convert_59class(label_400):
    """
    Load label image as 1 x height x width integer array of label indices.
    The leading singleton dimension is required by the loss.
    The full 400 labels are translated to the 59 class task labels.
    """
    labels_400 = [label.replace(' ','') for idx, label in np.genfromtxt('classes-400.txt', delimiter=':', dtype=None)]
    labels_59 = [label.replace(' ','') for idx, label in np.genfromtxt('classes-59.txt', delimiter=':', dtype=None)]
    for main_label, task_label in zip(('table', 'bedclothes', 'cloth'), ('diningtable', 'bedcloth', 'clothes')):
        labels_59[labels_59.index(task_label)] = main_label

    label = np.zeros_like(label_400, dtype=np.uint8)
    for idx, l in enumerate(labels_59):
        idx_400 = labels_400.index(l) + 1
        label[label_400 == idx_400] = idx + 1
    label = label[np.newaxis, ...]
    return label

path=os.path.join(os.getcwd(),'GroundTruth_trainval_mat') 
files=os.listdir(path)

labels_path=os.path.join(os.getcwd(),'GroundTruth_trainval_png')

for afile in files:
    file_path=os.path.join(path,afile)

    if os.path.isfile(file_path):
        if os.path.getsize(file_path)==0:
            continue
        mat_idx=afile[:afile.find('.mat')]
        mat_file=sio(file_path)
        label_400=np.array(mat_file['LabelMap'])
        
        mat_file59=PascalContext_400class_Convert_59class(label_400)
        mat_file59=mat_file59.astype(np.uint8)
        print(mat_file59.shape)
        label59_img=Image.fromarray(mat_file59.reshape(mat_file59.shape[1],mat_file59.shape[2]))
        dst_path=os.path.join(labels_path,mat_idx+'.png')
        print(dst_path)
        label59_img.save(dst_path)
