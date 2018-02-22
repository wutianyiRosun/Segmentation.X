### The PascalContext Dataset

PASCAL-Context provides whole scene annotations of PASCAL VOC 2010. While there are over 400 distinct classes, we follow the 59 class task that　picks the most frequent classes.The PASCAL-Context dataset provides detailed semantic labels for the whole scene, including both object (e.g., person) and stuff (e.g., sky). Following [The role of context for object detection and semantic segmentation in the wild,in CVPR, 2014], the proposed models are evaluated on the most frequent 59 classes along with one background category. The training set and validation set contain 4998 and 5105 images. 

train: 4998; validation: 5105张图片的标注 

Refer to `classes-59.txt` for the listing of classes in model output order.
Refer to `pascalcontext_layers.py` for the Python data layer for this dataset.
Note that care must be taken to map the raw class annotations into the 59 class task, as handled by our data layer.
See the dataset site: http://www.cs.stanford.edu/~roozbeh/pascal-context/


