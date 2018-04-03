# **deep_objdetect**
A framework for experiments on object detection using the darknet (yolo) and py-faster-rcnn techniques.

This repository uses modified forks of the [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) and [darknet](https://pjreddie.com/darknet/) implementations. It is intended to provide an interface which is easy to use to do quick experiments with different kinds of deep networks and different kinds of data as well. Just mention the directories in the master config file and run the code. Below are mentioned the basic requirements for getting started, make sure you have all this set up before you proceed with the build. Note that these are configurations which have been built successfully and other configs and setups may or may not work.

## Suggested OS and System Requirements:
Ubuntu - 16.04 / 14.04  
Linux in General would probably be fine.  
NVIDIA 1050/60/70 works fine(tested) - at least 4GB is good practically.  

## Pre-Installed Packages:
cuda v9.0 / v9.1  
cudnn v7.0.5 / v7.1.1(get the correct package based on which version of cuda is installed)  
gcc v5.4 / v5.2  
g++ v5.4 / v5.2

## Download all the source code properly
```
git clone --recursive https://github.com/amitfishy/deep_objdetect
```
OR if you just cloned the repo the normal way:
```
git clone https://github.com/amitfishy/deep_objdetect
git submodule update --init --recursive
```
## Some Other General Requirements for getting a successful build, with required cmd for Ubuntu:

### **TLDR:**
```
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libgoogle-glog-dev libgflags-dev liblmdb-dev libopenblas-dev libhdf5-serial-dev libprotobuf-dev protobuf-compiler libleveldb-dev libsnappy-dev libopencv-dev
sudo apt-get install python-dev python-setuptools python-numpy cython python-pip python-opencv python-skimage python-yaml
sudo pip install easydict protobuf
```

### More detailed information:
Generic Packages:
```
sudo apt-get install libleveldb-dev libsnappy-dev libopencv-dev
```
Python Stuff:
```
sudo apt-get install python-dev python-setuptools python-numpy cython python-pip python-opencv python-skimage python-yaml
sudo pip install easydict protobuf
```
Google Protobuf:
```
sudo apt-get install libprotobuf-dev protobuf-compiler
```
hdf5:
```
sudo apt-get install libhdf5-serial-dev
```
Boost:
```
sudo apt-get install --no-install-recommends libboost-all-dev
```
Gflags:
```
sudo apt-get install libgflags-dev
```
Glog:
```
sudo apt-get install libgoogle-glog-dev
```
lmdb:
```
sudo apt-get install liblmdb-dev
```
OpenBlas(for better CPU performance):
```
sudo apt-get install libopenblas-dev
```

## General Information
Note: `$PROJECT_ROOT` refers to this repository (path to `deep_objdetect` if you haven't changed anything)

1. After the installation of the basic dependencies is completed, follow the build instructions given [here](help/docs/BUILD_INSTRUCTIONS.md).

2. If you are running into issues during the build, please look [here](help/docs/GENERAL_BUILD_ISSUES.md). If this does not help please look at the original [py-faster-rcnn repo](https://github.com/rbgirshick/py-faster-rcnn) and [darknet repo](https://pjreddie.com/darknet/).

3. If the build was successfully tested, proceed to the [quick start guide](help/docs/GETTING_STARTED.md).

4. For an overview of the overall functionality, please look [here](help/docs/USAGE_HELP.md).

5. To do more detailed experiments with the deep networks and to change number of classes, have a look at [this](help/docs/EDITING_LOWER_LEVEL_CONFIGS.md).

6. If you want to extend the functionality (in the context of extending annotations and annotation filtering), please look [here](help/docs/EXTEND_FUNCTIONALITY.md).
Note: You probably won't need to do this.