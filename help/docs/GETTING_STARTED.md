# **Getting Started**

Follow these steps only if you were successfully able to complete the [build](BUILD_INSTRUCTIONS.md).
Firstly open up the master config file `$PROJECT_ROOT/objdet_experiments_conf.ini`

Under the section `[system_stuff]` set the `gpu_id` entry to what you got from the `./deviceQuery` CUDA built sample. Usually it is 0.

## 1. Test Online Detector with Pre-trained models

Ignore the section `[data_dirs]` for now. This step is just to see some pre-trained models in action as a python module.

### faster - rcnn

Under the `[faster_rcnn]` section go down to where `weights_file_online` entry is there.
While following the build instructions, you would have downloaded a weights folder called `faster_rcnn_models` with a weights file in it called `ZF_faster_rcnn_final.caffemodel`. Get the complete path for this file and put it under `weights_file_online` entry.
Just below this entry, you can also change the `detection_thresh_online` and `nms_thresh_online` to adjust the precision/recall of the detector.

Again below this add your actual `$PROJECT_ROOT` to `test_proto_file`, `class_names_file` and `cfg_file` so that they now point to the sample config data provided in this repository (the `$PROJECT_ROOT/config_data/online_detection_exp/py_faster_rcnn_configs` folder). 

**IMP: IF YOU DO EXPERIMENTS ON YOUR OWN ALWAYS MAKE SURE THE `test_proto_file`, `class_names_file` AND `cfg_file` REMAIN CONSISTENT WITH THE GIVEN `weights_file_online` OR YOU WILL GET UNEXPECTED OR ERRONEOUS OUTPUTS**

### darknet

Under the `[yolo]` section go down to where `weights_file_online` entry is there.
While following the build instructions, you would have downloaded a weights file called `yolo.weights`. Get the complete path for this file and put it under `weights_file_online` entry.
Just below this entry, you can also change the `detection_thresh_online` and `nms_thresh_online` to adjust the precision/recall of the detector.

Again below this add your actual `$PROJECT_ROOT` to `network_cfg_file` and `class_names_file` so that they now point to the sample config data provided in this repository (the `$PROJECT_ROOT/config_data/online_detection_exp/yolo_configs` folder).

**IMP: IF YOU DO EXPERIMENTS ON YOUR OWN ALWAYS MAKE SURE THE `network_cfg_file` AND `class_names_file` REMAIN CONSISTENT WITH THE GIVEN `weights_file_online` OR YOU WILL GET UNEXPECTED OR ERRONEOUS OUTPUTS**

-----------------------------------------------------------------------------

**Now we are done with the configuration for using the detector as a module.**

Have a look at how to use the module in the file `$PROJECT_ROOT/online_objdet_example.py`:
1. Initialize the module with the config file.
2. Initialize the online detector using the `faster_rcnn_online_init` or `yolo_online_init` member functions. This allocates GPU memory and holds it so that we can run the detector without re-initializing the deep networks every time.
3. Use the `faster_rcnn_online` and `yolo_online` member functions to process frames read in the OpenCV format.

Edit the simple script to add your own images to process (give absolute image paths to the `imagefiles` list in the python script)

Run this script using:
```
python online_objdet_example.py
```
This just uses the basic yolov1 network and the faster-rcnn with ZF backbone. In a similar way, you can try out yolov2 and faster-rcnn with a VGG16 or RESNET backbone for much better results. You might want to train and test over different datasets yourself, so look at the following section.

-----------------------------------------------------------------------------

## 2. Training Models from Scratch and Testing them out

Here we are going to train models over the 20 classes of the PASCAL VOC dataset. The basic yolov2 and faster-rcnn with a ZF backbone. Classes can be selectively chosen for training by following [this](EDITING_LOWER_LEVEL_CONFIGS.md). The parameters for training can be adjusted to make it converge faster. Otherwise starting training with a pre-trained model (like ImageNet models) is highly recommended and has shown the best results. I recommend training from scratch before training with the pre-trained ImageNet model to get a better feel of things.

First download the data [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#data).
Go to the development kit section and click on `Download the training/validation data (2GB tar file)`
Unzip it and keep it somewhere (separate from the code would be neater). You should get a folder called `VOCdevkit` with the following dir structure:
- VOC2012
	- Annotations
	- ImageSets
	- JPEGImages
	- Otherfolder1
	- Otherfolder2

We only need the `Annotations`, `ImageSets` and `JPEGImages`.

First go into the `ImageSets/Main` folder and delete everything **except** the `trainval.txt` and create an empty file called `test.txt`. Rename the `trainval.txt` into `train.txt`. Note that we would want to be flexible with our train set and validation set, so we merge it into one file and later divide it randomly based on what is required. It is also important to name the files `train.txt` (the one for which annotations are present) and `test.txt` (the one for which annotations are not present, but detection outputs are needed).

Now we are going to add the relevant directories of this dataset to our framework so we can train and test it over a validation set. Go the `$PROJECT_ROOT/objdet_experiments_conf.ini` and edit the `[data_dirs]` section.
```
image_data_dir: $DATA_DOWNLOAD_DIR/VOCdevkit/VOC2012/JPEGImages
annotation_dir: $DATA_DOWNLOAD_DIR/VOCdevkit/VOC2012/Annotations
image_id_dir: $DATA_DOWNLOAD_DIR/VOCdevkit/VOC2012/ImageSets/Main
proj_root_dir: $PROJECT_ROOT
```

Now move on to the `[faster_rcnn]` section.

### faster - rcnn

Set `use_faster_rcnn` to `True`. This should be done when we want to run `train` or `test` experiments with datasets, it is not required for the online detector (irrelevant in that case). Let's call our experiment `experiment_voc20`, so that the outputs are formatted neatly. Mention where you want the outputs using `output_directory`. You can use anything for `dataset_type` (don't use `kitti`, it is a special keyword - refer to [this](USAGE_HELP.md)). Set the `train_mode` and `test_mode` to 1 to include all the data in the annotations (this is only relevant when using `kitti`).
```
use_faster_rcnn: True
experiment_name: experiment_voc20
output_directory: $YOUR_OUTPUT_DIR/pyfrcnn
dataset_type: pascalvoc2012

train_mode: 1
test_mode: 1
```

Set `use_validation_experiments` to `False`. This will generate a fixed split of train-val data dictated by the `train_data_fraction` parameter. We can indicate whether we want to `train` or `test` by setting these to `True` or `False`. We can indicate the `train_split` as `train` or `val` and `test_split` as `val` or `test`. Evaluate should be set to `True`.
```
use_validation_experiments: False
train: True
train_split: train
test: True
test_split: val
evaluate: True
```

Note that if you set `use_validation_experiments` to `True`, it will generate a number of random splits given by `num_repetitions` and evaluate over each of these separate splits to get a distribution of mAP results. In this case `test_split` should not be `test`, because we generally assume that we don't have the annotations for the `test` `test_split`. We also do not store any models in this case because it takes up a lot of space for each experiments by a factor of `num_repetitions`. These experiments are only meant to get the mAP by taking random splits and giving the overall result, so you can find the overall mean/variance of the algorithm over a dataset etc. However if you want the model and you want to keep your data as a fixed split, then do not use this.
```
train_data_fraction: 0.8
num_repetitions: 10			//irrelevant if `use_validation_experiments: False`
```

You can use the 11-point AP metric (2007) or the total area of the AP curve with `use07metric`. Set it to `False` to use the area metric. Set the number of iterations using `num_iterations`, batch size and other such training related parameters can be adjusted in the `solver.prototxt` and the `config.yaml` files. `network_name` is not actually used right now, but you can add it for your reference. Model names can be modified using `output_model_prefix`.
```
use07metric: False
num_iterations: 100000
network_name: ZF
output_model_prefix: frcnn_zf_voc2012
```

You don't have to start from scratch, this is being done to get a better understanding of the whole process. It is highly recommended to start with a pre-trained ImageNet model, but try it from scratch initially. Set the `use_pretrained_weights` to `False` if you want to start from scratch. If you set it to `True`, you should add the pre-trained model using `pretrained_weights_file`. If you want to run the model you just trained over the test split then set `use_trained_weights_test` to `True`. Otherwise for stand alone testing experiments set it to `False` and add the file to `weights_file_test`.
```
use_pretrained_weights: False
pretrained_weights_file: IDONTHAVETHISFILE

use_trained_weights_test: True
weights_file_test: IDONTHAVETHISFILE
```

We are going to use the end to end (joint) method of training. So we use the corresponding prototxt files for a ZF network end to end with the 20 PASCAL VOC classes. A list of class names should also be provided under `class_names_file` for whatever categories need to be trained. You can leave out certain names here and data will be taken accordingly, however you must always use the same names file with the same proto files (train and test), same cfg file and weight files.

The files for this experiment have been provided in `$PROJECT_ROOT/config_data/train_test_exp_20/py_faster_rcnn_configs` so just add those along to the correct places as shown:
```
solver_proto_file: $PROJECT_ROOT/config_data/train_test_exp_20/py_faster_rcnn_configs/solver.prototxt
train_proto_file: $PROJECT_ROOT/config_data/train_test_exp_20/py_faster_rcnn_configs/train.prototxt
test_proto_file: $PROJECT_ROOT/config_data/train_test_exp_20/py_faster_rcnn_configs/test.prototxt
class_names_file: $PROJECT_ROOT/config_data/train_test_exp_20/py_faster_rcnn_configs/voc.names
cfg_file: $PROJECT_ROOT/config_data/train_test_exp_20/py_faster_rcnn_configs/config.yml
```

Now move on to the `[yolo]` section.

### darknet

Most of the settings are common to that of faster - rcnn:
```
use_yolo: True
experiment_name: experiment_voc20
output_directory: $YOUR_OUTPUT_DIR/darknet
dataset_type: pascalvoc2012

train_mode: 1
test_mode: 1

use_validation_experiments: False
train: True
train_split: train
test: True
test_split: val
evaluate: True

train_data_fraction: 0.8
num_repetitions: 10			//irrelevant if `use_validation_experiments: False`

use07metric: False
num_iterations: 10000
network_name: darknet19v2
output_model_prefix = yolov2_voc2012

use_pretrained_weights: False
pretrained_weights_file: IDONTHAVETHISFILE

use_trained_weights_test: True
weights_file_test: IDONTHAVETHISFILE
```

Again here set the number of iterations using `num_iterations`, batch size and other such training related parameters can be adjusted in the `network_cfg_file` file. `network_name` is not actually used right now, but you can add it for your reference.

Now we are going to use yolov2 for training. So we use the corresponding `network_cfg_file` with the number of classes as 20 (car). A list of class names should also be provided under `class_names_file` for whatever categories need to be trained. You can leave out certain names here and data will be taken accordingly, however you must always use the same names file with the same `network_cfg_file` and weight files.

The config files for this experiment have been provided in `$PROJECT_ROOT/config_data/train_test_exp_20/yolo_configs` so just add those along to the correct places as shown:
```
network_cfg_file: $PROJECT_ROOT/config_data/train_test_exp_20/yolo_configs/yolo-voc.cfg
class_names_file: $PROJECT_ROOT/config_data/train_test_exp_20/yolo_configs/voc.names
```

After this go to `$PROJECT_ROOT` and run `objdet_experiments.py` with:
```
python objdet_experiments.py
```

After it's done, check out the results where you specified the output folder. There will be a `models` folder which was generated during training and a `results` folder which was generated using the model from the `models` folder. In the results folder you have detection files in the standard PASCAL VOC test output (i.e. for submission). Also if you set `evaluate` to `True` and you had the annotations for the `test_split`, then you will get a file containing the average precision for each class and the mean AP as well.

You can try using the obtained models with the online detector to see some live results from your images (in the same way you used the downloaded models in the first part of this doc). Do not expect great results! For better results try training over an ImageNet pre-trained model.

-----------------------------------------------------------------------------

## 3. Training Models starting from Pre-Trained ImageNet Models and Testing them out

This part covers how you can begin to train from a given starting point (i.e a pretrained model). Firstly, let's reuse the same `$PROJECT_ROOT/objdet_experiments_conf.ini` file after making a few small changes. We earlier did an experiment training from scratch, whose config files are in `train_test_exp_20`. Similarly the config files we want to use for the ImageNet initilization are in `train_test_exp_20_imagenet_init`.

So wherever it occurs change `train_test_exp_20` to `train_test_exp_20_imagenet_init`. Note that they are exactly the same, but we keep them separately because we will need to make minor changes later on so that we can load our downloaded ImageNet models (and later on revert those changes) without messing with something that is already working fine.

Also change `experiment_name` to `experiment_voc20_inet` to format the results in a different folder. Also initially set the `test` flag to `False` in both `[yolo]` and `[faster_rcnn]` since initially we will only be getting the pre-trained models and we would like to test them out only after we fine-tune it.

First download the pre-trained ZF ImageNet model by MSR as follows.
```
cd dl_algos/py_faster_rcnn
./data/scripts/fetch_imagenet_models.sh 
```
You will get some pre-trained ImageNet models in `$PROJECT_ROOT/dl_algos/py_faster_rcnn/data/imagenet_models`. Rename `ZF.caffemodel` to `ZF_0.caffemodel` to mark that this the starting point of our training at iteration #0.

To alter the structure of the ZF weights model from the classification network to the object detection network, we need to edit the `$PROJECT_ROOT/config_data/train_test_exp_20_imagenet_init/py_faster_rcnn_configs/train.prototxt`. If a weights model is loaded based on a prototxt file, then the sequence of events is as follows:
- Network is initialized based on the details provided in the proto file
- Layer names are matched between the prototxt and the weight files and weights are loaded into their respective layers (overwriting any initialization provided by the proto file)
- Any layer names present in the weights file which are not there in the proto definition are ignored

So a good way to convert the weights file from a object classification to an object detection format is to rename all the layers that we **DO NOT** want to load, into something else (eg: make 'fc6' -> 'fc6abcdef' so it will be ignored from weights file). So in this case, we want everything upto layer 5 to be loaded. Rename all the layers after that by adding some simple string (like 'abcd'). Note that it is not necessary to rename all the layers after that, but only the ones which we do not want to load (if they happen to have a matching name in the weights file). It is generally simpler to just rename everything that you do not want to load just in case. Now what we can do is set the `num_iterations` to 0, so that we just get an output model in the object detection format whose new layers are randomly initialized.

Set `use_yolo: False` under the `[yolo]` section for this part.

Now move on to the `[faster_rcnn]` section.

```
num_iterations: 0
use_pretrained_weights: True
pretrained_weights_file: $PROJECT_ROOT/dl_algos/py_faster_rcnn/data/imagenet_models/ZF_0.caffemodel
```

Run this to generate the intermediate weights file by using the following:
```
cd ../../			//go to $PROJECT_ROOT
python objdet_experiments.py
```

We also need to rename the layers back (since some of the code expects certain names for the layers), so revert back all the changed layer names to what was there originally in the `train.prototxt` file. We generated an intermediate weights file which we use as `pretrained_weights_file` and overwrite it to get the required file by doing the following:
```
num_iterations: 0
use_pretrained_weights: True
pretrained_weights_file: $YOUR_OUTPUT_DIR/pyfrcnn/experiment_voc20_inet/models/80-20/frcnn_zf_voc2012_0.caffemodel
```
Run the script again to get the proper format weights file:
```
python objdet_experiments.py
```
This will overwrite the older file and give us the new one in the proper and expected format. Now we can train this in the same way as we did earlier, except that we provide the `pretrained_weights_file`
Keeping other settings the same, let us change the following to train over an ImageNet initialization:
```
test: True
num_iterations: 100000
use_pretrained_weights: True
pretrained_weights_file: $YOUR_OUTPUT_DIR/pyfrcnn/experiment_voc20_inet/models/80-20/frcnn_zf_voc2012_0.caffemodel
```





Set `use_faster_rcnn: False` under the `[faster_rcnn]` section and set `use_yolo: True` under the `[yolo]` section for this part.

Here we follow a slightly different procedure but the concept is the same. The cfg file that is provided has a default initialization. If no weights file is provided then the random weights based on the cfg file contents are used as a starting point. If however, a weights file is provided, then **AFTER** the random initialization, the random weights of the network are overwritten by the weights in the weights file. All we need to do is to make sure that the weights file has the same structure as the cfg file (upto any point in the cfg file).

First we download the darknet19 high res model (448 by 448) which was fine-tuned over the ImageNet data.
```
cd dl_algos/darknet
wget https://pjreddie.com/media/files/darknet19_448.weights
```

Then we chunk out the portion we need for training our yolov2 object detection model using the `./darknet partial` API.
```
./darknet partial cfg/darknet19_448.cfg darknet19_448.weights darknet19v2_0.weights 23			//we use these weights upto 23 layers
```
The output file name is `darknet19v2_0.weights`. It is again important to indicate the `_0` in the end of the filename before the `.weights` extension to indicate that this is the starting point for training (the 0th iteration).

Now move on to the `[yolo]` section.

```
num_iterations: 0
use_pretrained_weights: True
pretrained_weights_file: $PROJECT_ROOT/dl_algos/darknet/darknet19v2_0.weights
```

Run this to generate the initial weights file by using the following:
```
cd ../../			//go to $PROJECT_ROOT
python objdet_experiments.py
```
Note that here the re-naming procedure is not required because the `./darknet partial` API makes it easier for us.

Now we've got our intial starting point at `$YOUR_OUTPUT_DIR/darknet/experiment_voc20_inet/models/80-20/yolov2_voc2012_0.weights`. Keeping other settings the same, let us change the following to do a small amount of training over an ImageNet initialization:
```
test: True
num_iterations: 10000
use_pretrained_weights: True
pretrained_weights_file: $YOUR_OUTPUT_DIR/darknet/experiment_voc20_inet/models/80-20/yolov2_voc2012_0.weights
```

Set `use_faster_rcnn: True` under the `[faster_rcnn]` section and set `use_yolo: True` under the `[yolo]` section.
Then run both the experiments the standard way using:
```
python objdet_experiments.py
```
Observe how we get better results using the same training settings. That is one of the benefits of using pre-trained models as a starting point.