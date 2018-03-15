#the script extract 60 color from color150 of ADE Dataset

import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np  

b=sio.loadmat('color150.mat')
print(type(b['colors']))
print((b['colors']).shape)
color60=b['colors'][:60]
print(color60.shape)
author= 'Rosun'
sio.savemat('color60.mat',{'color60': color60, 'author': author})
