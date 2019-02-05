# Awesome Semantic Segmentation

## Networks by architecture
### Semantic segmentation
- U-Net [https://arxiv.org/pdf/1505.04597.pdf]
	+ https://github.com/ZijunDeng/pytorch-semantic-segmentation [PyTorch]
- SegNet [https://arxiv.org/pdf/1511.00561.pdf]
- DeepLab [https://arxiv.org/pdf/1606.00915.pdf]
	+ https://github.com/isht7/pytorch-deeplab-resnet [PyTorch]
	+ https://github.com/bermanmaxim/jaccardSegment [PyTorch]
- FCN [https://arxiv.org/pdf/1605.06211.pdf]
	+ https://github.com/wkentaro/pytorch-fcn [PyTorch]
	+ https://github.com/ycszen/pytorch-seg [PyTorch]
	+ https://github.com/Kaixhin/FCN-semantic-segmentation [PyTorch]
- ENet [https://arxiv.org/pdf/1606.02147.pdf]
- LinkNet [https://arxiv.org/pdf/1707.03718.pdf]
- DenseNet [https://arxiv.org/pdf/1608.06993.pdf]
- Tiramisu [https://arxiv.org/pdf/1611.09326.pdf]
- DilatedNet [https://arxiv.org/pdf/1511.07122.pdf]
- PixelNet [https://arxiv.org/pdf/1609.06694.pdf]
- ICNet [https://arxiv.org/pdf/1704.08545.pdf]
- ERFNet [http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf]
- RefineNet [https://arxiv.org/pdf/1611.06612.pdf]
- PSPNet [https://arxiv.org/pdf/1612.01105.pdf,https://hszhao.github.io/projects/pspnet/]
- Dilated convolution [https://arxiv.org/pdf/1511.07122.pdf]
	+ https://github.com/fyu/drn#semantic-image-segmentataion [PyTorch]
	+ https://github.com/hangzhaomit/semantic-segmentation-pytorch [PyTorch]
- DeconvNet [https://arxiv.org/pdf/1505.04366.pdf]
- FRRN [https://arxiv.org/pdf/1611.08323.pdf]
- GCN [https://arxiv.org/pdf/1703.02719.pdf]
	+ https://github.com/ycszen/pytorch-seg [PyTorch]
- LRR [https://arxiv.org/pdf/1605.02264.pdf]
- DUC, HDC [https://arxiv.org/pdf/1702.08502.pdf]
	
- MultiNet [https://arxiv.org/pdf/1612.07695.pdf]
	+ https://github.com/MarvinTeichmann/MultiNet
	+ https://github.com/MarvinTeichmann/KittiSeg
- Segaware [https://arxiv.org/pdf/1708.04607.pdf]
- Semantic Segmentation using Adversarial Networks [https://arxiv.org/pdf/1611.08408.pdf]
- PixelDCN [https://arxiv.org/pdf/1705.06820.pdf]
	
### Instance aware segmentation
- FCIS [https://arxiv.org/pdf/1611.07709.pdf]
	
- MNC [https://arxiv.org/pdf/1512.04412.pdf]
	
- DeepMask [https://arxiv.org/pdf/1506.06204.pdf]
	
- SharpMask [https://arxiv.org/pdf/1603.08695.pdf]
	
- Mask-RCNN [https://arxiv.org/pdf/1703.06870.pdf]
	
- RIS [https://arxiv.org/pdf/1511.08250.pdf]
  
- FastMask [https://arxiv.org/pdf/1612.08843.pdf]
 
- BlitzNet [https://arxiv.org/pdf/1708.02813.pdf]

### Weakly-supervised segmentation
- SEC [https://arxiv.org/pdf/1603.06098.pdf]
  
## RNN
- ReNet [https://arxiv.org/pdf/1505.00393.pdf]
 
- ReSeg [https://arxiv.org/pdf/1511.07053.pdf]
  + https://github.com/Wizaron/reseg-pytorch [PyTorch]
  
- RIS [https://arxiv.org/pdf/1511.08250.pdf]
 
- CRF-RNN [http://www.robots.ox.ac.uk/%7Eszheng/papers/CRFasRNN.pdf]
  

## Graphical Models (CRF, MRF)
  + https://github.com/cvlab-epfl/densecrf
  + http://vladlen.info/publications/efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials/
  + http://www.philkr.net/home/densecrf
  + http://graphics.stanford.edu/projects/densecrf/
  + https://github.com/amiltonwong/segmentation/blob/master/segmentation.ipynb
  + https://github.com/jliemansifry/super-simple-semantic-segmentation
  + http://users.cecs.anu.edu.au/~jdomke/JGMT/
  + https://www.quora.com/How-can-one-train-and-test-conditional-random-field-CRF-in-Python-on-our-own-training-testing-dataset
  + https://github.com/tpeng/python-crfsuite
  + https://github.com/chokkan/crfsuite
  + https://sites.google.com/site/zeppethefake/semantic-segmentation-crf-baseline
  + https://github.com/lucasb-eyer/pydensecrf

## Datasets:

  + [Stanford Background Dataset](http://dags.stanford.edu/projects/scenedataset.html)
  + [Sift Flow Dataset](http://people.csail.mit.edu/celiu/SIFTflow/)
  + [Barcelona Dataset](http://www.cs.unc.edu/~jtighe/Papers/ECCV10/)
  + [Microsoft COCO dataset](http://mscoco.org/)
  + [MSRC Dataset](http://research.microsoft.com/en-us/projects/objectclassrecognition/)
  + [LITS Liver Tumor Segmentation Dataset](https://competitions.codalab.org/competitions/15595)
  + [KITTI](http://www.cvlibs.net/datasets/kitti/eval_road.php)
  + [Pascal Context](http://www.cs.stanford.edu/~roozbeh/pascal-context/)
  + [Data from Games dataset](https://download.visinf.tu-darmstadt.de/data/from_games/)
  + [Human parsing dataset](https://github.com/lemondan/HumanParsing-Dataset)
  + [Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas)
  + [Microsoft AirSim](https://github.com/Microsoft/AirSim)
  + [MIT Scene Parsing Benchmark](http://sceneparsing.csail.mit.edu/)
  + [COCO 2017 Stuff Segmentation Challenge](http://cocodataset.org/#stuff-challenge2017)
  + [ADE20K Dataset](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
  + [INRIA Annotations for Graz-02](http://lear.inrialpes.fr/people/marszalek/data/ig02/)
  + [Daimler dataset](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/daimler_pedestrian_benchmark_d.html)
  + [ISBI Challenge: Segmentation of neuronal structures in EM stacks](http://brainiac2.mit.edu/isbi_challenge/)
  + [INRIA Annotations for Graz-02 (IG02)](https://lear.inrialpes.fr/people/marszalek/data/ig02/)
  + [Pratheepan Dataset](http://cs-chan.com/downloads_skin_dataset.html)
  + [Clothing Co-Parsing (CCP) Dataset](https://github.com/bearpaw/clothing-co-parsing)
  + [Inria Aerial Image](https://project.inria.fr/aerialimagelabeling/)

## Benchmarks
  + https://github.com/ZijunDeng/pytorch-semantic-segmentation [PyTorch]
  + https://github.com/meetshah1995/pytorch-semseg [PyTorch]
 
## Annotation Tools:

  + https://github.com/AKSHAYUBHAT/ImageSegmentation
  + https://github.com/kyamagu/js-segment-annotator
  + https://github.com/CSAILVision/LabelMeAnnotationTool
  + https://github.com/seanbell/opensurfaces-segmentation-ui
  + https://github.com/lzx1413/labelImgPlus
  + https://github.com/wkentaro/labelme


## Metrics
  + https://github.com/martinkersner/py_img_seg_eval
  
## Other lists
  + https://github.com/tangzhenyu/SemanticSegmentation_DL
  + https://github.com/nightrome/really-awesome-semantic-segmentation
  
## Medical image segmentation:

- DIGITS
  + https://github.com/NVIDIA/DIGITS/tree/master/examples/medical-imaging
  
  
- Cascaded-FCN
  + https://github.com/IBBM/Cascaded-FCN
  

  

  

- Papers:
  + https://www2.warwick.ac.uk/fac/sci/dcs/people/research/csrkbb/tmi2016_ks.pdf
  + Sliding window approach
	  - http://people.idsia.ch/~juergen/nips2012.pdf
  + https://github.com/albarqouni/Deep-Learning-for-Medical-Applications#segmentation
	  
 - Data:
   - https://luna16.grand-challenge.org/
   - https://camelyon16.grand-challenge.org/
   - https://github.com/beamandrew/medical-data
  


## Video segmentation

  + https://github.com/shelhamer/clockwork-fcn
  + https://github.com/JingchunCheng/Seg-with-SPN



### Other



## Papers and Code (Older list)

- Simultaneous detection and segmentation

  + http://www.eecs.berkeley.edu/Research/Projects/CS/vision/shape/sds/
  + https://github.com/bharath272/sds_eccv2014
  
- Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation

  + https://github.com/HyeonwooNoh/DecoupledNet
  
- Learning to Propose Objects

  + http://vladlen.info/publications/learning-to-propose-objects/ 
  + https://github.com/philkr/lpo
  
- Nonparametric Scene Parsing via Label Transfer

  + http://people.csail.mit.edu/celiu/LabelTransfer/code.html
  

  
## To look at
  + https://github.com/fchollet/keras/issues/6538
  + https://github.com/warmspringwinds/tensorflow_notes
  + https://github.com/kjw0612/awesome-deep-vision#semantic-segmentation
  + https://github.com/desimone/segmentation-models
  + https://github.com/nightrome/really-awesome-semantic-segmentation
  + https://github.com/kjw0612/awesome-deep-vision#semantic-segmentation
  + http://www.it-caesar.com/list-of-contemporary-semantic-segmentation-datasets/
  + https://github.com/MichaelXin/Awesome-Caffe#23-image-segmentation


## Blog posts, other:

  + https://handong1587.github.io/deep_learning/2015/10/09/segmentation.html
  + http://www.andrewjanowczyk.com/efficient-pixel-wise-deep-learning-on-large-images/
  + https://devblogs.nvidia.com/parallelforall/image-segmentation-using-digits-5/
  + https://github.com/NVIDIA/DIGITS/tree/master/examples/binary-segmentation
  + https://github.com/NVIDIA/DIGITS/tree/master/examples/semantic-segmentation

