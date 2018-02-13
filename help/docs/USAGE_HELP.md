# **Usage Help**

To use this work properly, follow these guidelines. Before following this, make sure the demo runs successfully for darknet and py_faster_rcnn after following the build instructions [here](BUILD_INSTRUCTIONS.md).

There are 2 main components to use in this repo:
1. Object Detection Training and Testing Framework.
2. Online Detector for use in your own code.

The configuration parameters for both these cases are controlled through a [config file](../../objdet_experiments_conf.ini). This file is divided into various sections for different purposes and has fixed headers for each setting (look at the file for an example).

As of now there are 4 sections:

## 'system_stuff' : Mention system parameters here.
-'gpu_id' : Provide gpu id here(No multi-gpu support)

## 'data_dirs' : Mention particular of dataset here:
-'image_data_dir' : Should contain all images here(test and train) :- only formats (.jpg, .png, .jpeg)
-'annotation_dir' : This folder should have one xml file for each image in 'image_data_dir' in the standard PASCAL format. 'DontCare' boxes can be included in the annotations to ignore during evaluation. However multiple detections on 'DontCare' boxes will count as false positives.
-'image_id_dir' : This folder should contain 2 files :- 'train.txt' and 'test.txt', which list out the basename (no extension) of each image file to be considered during testing and training. Note that an error will pop-up if an entry given in 'train.txt' has either no image file or annotation file. However an entry in 'test.txt' which has an image file but no annotation file will not give an error. A detection output will be given in this case without evaluating the mean Average Precision.
-'proj_root_dir' : Mention the location of the project root folder you stored the repo in.

## 'faster_rcnn' : Mention Parameters for faster rcnn here

### Experiments With Detection
-'use_faster_rcnn' : You can set this to 'False' to skip its usage for training and testing (you can still use the online detector though)
-'experiment_name' : This helps in formatting the output.
-'output_directory' : Outputs are stored here. Outputs on the training side consist of trained model files, on the testing side it consists of giving output detection info in a file (PASCAL format) and evaluating mAP if the annotations are available and storing this in a file.
-'dataset_type' : Additional data filtering criteria can be used by setting the experiment type, both for training and testing. For example look at the (kitti-2d benchmark)[http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d], which defines easy, moderate and hard levels based on additional annotation information in images. Put this field to 'kitti' to use those benchmarks. In most cases, just use any other name (generic) to include all the data.
-'train_mode' : This sets the level at which the training data is filtered. The levels are defined by additional annotation info (take a look at 'kitti' - easy, moderate, hard). If the 'dataset_type' is 'kitti' then 1:easy, 2:moderate, 3:hard, 4:None.
-'test_mode' : Similar to the 'train_mode', the level of 'test_mode' can be set to restrict evaluation to a certain level. However multiple detections on higher levels will be counted as false positives (just like 'DontCare' boxes).
Please note that additional modes can be set for different 'dataset_type' by modifying the source code. Refer to [this](EXTEND_FUNCTIONALITY.md). Also keep in mind that each lower level should be a subset of the higher level
-'use_validation_experiments' : This is used for generating different 'train' and 'val' splits from the 'train.txt' image ID file. If this is 'True', the experiment is repeated 'num_repetitions' times with a trainval split (random each time) indicated by 'train_data_fraction'. To use the validation experiments 'train', 'test' and 'evaluate' should be set to 'True', 'train_split' = train and 'test_split' = val.
Keep 'use_validation_experiments' as 'False' if you want to keep the data fixed with a split indicated by 'train_data_fraction'. 'num_repetitions' becomes irrelevant in this case. Training without testing can be done in this case by setting 'test' to 'False'. Similarly testing without training can be done by keeping 'train' as 'False'.
-'use07metric' : Use the 11 Point metric for AP or the Area Under the AP curve by setting this to 'False'
-'num_iterations' : Indicates how many iterations are done (where each iteration is specified by the config file for py_faster_rcnn under 'cfg_file'). (This setting overrides the number of iterations provided in 'cfg_file' and 'proto_files')
-'pretrained_weights_file' : Begin training from a pre-trained weights file specified here (used only when 'use_pretrained_weights' is 'True'), otherwise trains from scratch
-'weights_file_test' : Specify a weights file for running Tests (only when 'use_trained_weights_test' is 'False'), otherwise use the file obtained just after training is completed (Note that 'train' should be 'True' in this case)
-'solver_proto_file', 'train_proto_file', 'test_proto_file' :- These mention details about the networks used for py_faster_rcnn. The weights files specify the numbers to use as weights, whereas these files describe how the network is structured so that the weights can be loaded onto them.
-'class_names_file' : Mention the class_names to be trained from the annotation files here. Make sure you use the same file during testing. This file contains each class name line by line (order should not be changed).

### Online Detection
Online Detection means using the doing detection on the spot, by using this codebase as a module. An example is provided on how to use this feature on your own code. This allows easy flexibility to use object detection in whatever way you want.
-'weights_file_online' : This specifies the weights file to use for the detection
-'detection_thresh_online' : This specifies the detection threshold to use. A higher threshold indicates higher 'precision' wheras a lower threshold indicates higher 'recall'. Set this to some value between 0 and 1 with the trade-off in mind.
-'nms_thresh_online' : This specifies the non-maximum suppression threshold to use. This helps to reduce the number of detections in a single location (thereby helping to reduce multiple activations on a single object). Generally keep this value low.

**IMP: WHEN DOING ANY EXP WITH A TRAINED MODEL, MAKE SURE THAT YOU USE THE SAME 'class_names_file', 'any_proto_file' AND 'cfg_file' FILES AS WHAT WAS USED TO GET THE TRAINED MODEL IN THE FIRST PLACE. ALWAYS MAKE SURE THAT THE FILES ARE CONSISTENT.**

## 'yolo' : Mention Parameters for faster rcnn here
All the parameters mentioned in 'faster_rcnn' are the same here as well. (Note: Evaluation is done with the same 'faster-rcnn' script to provide more meaningful results between algorithms).
-'network_cfg_file' : This file is similar to the 'proto' files in 'faster-rcnn'. It defines the network structure and other important network parameters as well some settings used in training.

**IMP: WHEN DOING ANY EXP WITH A TRAINED MODEL, MAKE SURE THAT YOU USE THE SAME 'class_names_file' AND 'network_cfg_file' FILES AS WHAT WAS USED TO GET THE TRAINED MODEL IN THE FIRST PLACE. ALWAYS MAKE SURE THAT THE FILES ARE CONSISTENT.**