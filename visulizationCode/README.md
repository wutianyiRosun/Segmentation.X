1.Visulize segmentation results.

ADE20K:         color150.mat     objectName150.mat
VOC 2012:       colormapvoc.mat  objectName21.mat
Cityscapes:     colormapcs.mat   objectName19.mat
PASCAL-Context: color60.mat      objectName60.mat
//
2. 由于该数据集并未提供官方的colormap和objectName. 利用gen_Pascal_Context_colormap.py, gen_Pascal_Context_objectName.m 分别生成PASCAL_Context数据集的colormap和objectName. 
