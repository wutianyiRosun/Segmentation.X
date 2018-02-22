
import os
from scipy.io import loadmat as sio
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

#path='/home/dl/DL_dataset/VOCdevkit/trainval'
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
        mat_file=np.array(mat_file['LabelMap'])
        #print mat_file.keys()
        mat_file=mat_file.astype(np.uint8)
        label_img=Image.fromarray(mat_file.reshape(mat_file.shape[0],mat_file.shape[1]))
        dst_path=os.path.join(labels_path,mat_idx+'.png')
        print(dst_path)
        label_img.save(dst_path)

