# **Editing Lower Level Config Files**

## darknet

-Edit the class names file for every class that you want to be trained over (information should be available as xml files for these classes).
-Edit the cfg given in config_data folder (this is for yolov1):
1. In the end of the file in section [region] change classes:ENTER NUMBER OF CLASSES HERE
2. In the end of the file in the last [convolutional] section change filters:(NUMBER OF CLASSES + 5) x 5
For example if you have 3 classes:
'''
.
.
[convolutional]
filters = 40
.
.

[region]
classes = 3
.
.
'''

You can experiment with the number and kind of layers, it is not very straightforward to explain but easier to understand by examples. If you've read the paper for [yolov1](https://arxiv.org/abs/1506.02640) or [yolov2](https://arxiv.org/abs/1612.08242), try to correlate the network they describe with the meta data in the cfg file.

There are some sample config files (in '$PROJECT_ROOT/config_data/experimentX/darknet') to give an idea of the design involved, which goes together with the yolo.weights file you downloaded to test the build(refer to [this](BUILD_INSTRUCTIONS.md)). This weights file detects bounding boxes from categories from the [COCO dataset](http://cocodataset.org/#home).
Have a look at the mentioned config folder '$PROJECT_ROOT/config_data/experimentX/darknet':
-The 'coco.names' file lists the categories involved line by line.
-The 'cfgs/yolo.cfg' file has the meta data for the network and training and testing parameters.

**IMP NOTE: For initial training, only the desired categories can be put into the .names file (the same number of classes should reflect in the .cfg files as well), and only the desired categories will be trained. HOWEVER IF USING SOME PRETRAINED WEIGHTS FILE, ALWAYS MAKE SURE THAT THE .NAMES FILE HAS THE CATEGORIES OF THAT WEIGHTS FILE ITSELF, AND THE .CFG FILES REFLECT THE NUMBER OF CLASSES OF THE WEIGHTS FILE. IN SHORT MAINTAIN CONSISTENCY BETWEEN THE WEIGHTS FILE, THE .NAMES FILE AND THE .CFG FILES**


## py_faster_rcnn

-Edit the class names file for every class that you want to be trained over (information should be available as xml files for these classes).
-Edit the prototxt files given in the config_data folder (this is for faster-rcnn with ZF backbone):
1. train.prototxt - Edit in 4 places as follows:
	-Add (number of classes + 1) where the 'num_classes' field is specified [2 places]
	-Towards the end of the file, find the layer with the name 'cls_score' and add (number of classes + 1) to the 'num_output' field [1 place]
	-Towards the end of the file, find the layer with the name 'bbox_pred' and add (number of classes + 1) x 4 to the 'num_output' field [1 place]
2. test.prototxt - Edit in 2 places as follows:
	-Towards the end of the file, find the layer with the name 'cls_score' and add (number of classes + 1) to the 'num_output' field [1 place]
	-Towards the end of the file, find the layer with the name 'bbox_pred' and add (number of classes + 1) x 4 to the 'num_output' field [1 place]
3. solver.prototxt - Can be changed for tuning the training settings. This can be left at the default values also.
4. config.yml - Can be changed for tuning the training and testing settings (like number of rpn proposals to consider etc.). This can be left at the default values also.

You can experiment with the number and kind of layers, it is not very straightforward to explain but easier to understand by examples. If you've read the paper for [faster-rcnn](https://arxiv.org/abs/1506.01497), try to correlate the networks they describe with the meta data in the train.prototxt and test.prototxt and config.yml (you can't really see the list of parameters here, the default settings for it are given in '$PROJECT_ROOT/dl_algos/py_faster_rcnn/lib/fast_rcnn/config.py' - you can affect all these parameters directly through the 'config.yml' file). 'solver.ptototxt' is useful for setting the training parameters.

There are some sample config files (in '$PROJECT_ROOT/config_data/experimentX/py_faster_rcnn') to give an idea of the design involved, which goes together with the faster_rcnn ZF model file you downloaded to test the build(refer to [this](BUILD_INSTRUCTIONS.md)). This weights file detects bounding boxes from categories from the [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html).
Have a look at the mentioned config folder '$PROJECT_ROOT/config_data/experimentX/py_faster_rcnn':
-The 'voc.names' file lists the categories involved line by line.
-The 'models/solver.prototxt' file has the general parameters for training the deep network.
-The 'models/test.prototxt' file defines the overall network structure.
-The 'models/train.prototxt' file defines the overall network structure but also has some information regarding training.
-The 'cfgs/config.yml' file indicates some very specific training and testing parameters

**IMP NOTE: For initial training, only the desired categories can be put into the .names file (the same number of classes should reflect in the .prototxt files as well), and only the desired categories will be trained. HOWEVER IF USING SOME PRETRAINED WEIGHTS FILE, ALWAYS MAKE SURE THAT THE .NAMES FILE HAS THE CATEGORIES OF THAT WEIGHTS FILE ITSELF, AND THE PROTOTXT FILES REFLECT THE NUMBER OF CLASSES OF THE WEIGHTS FILE. IN SHORT MAINTAIN CONSISTENCY BETWEEN THE WEIGHTS FILE, THE .NAMES FILE AND THE TRAIN/TEST PROTOTXT FILES**