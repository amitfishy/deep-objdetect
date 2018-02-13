
### **Build Instructions**

The only build required is for the py_faster_rcnn and darknet modules.

1. py_faster_rcnn Build

Go into the '$PROJECT_ROOT/dl_algos/py_faster_rcnn/caffe-fast-rcnn' folder and merge the updated caffe repo to allow cudnn v7 and cuda v9 to work:
'''
cd dl_algos/py_faster_rcnn/caffe-fast-rcnn
git remote add caffe https://github.com/amitfishy/caffe
git fetch caffe
git merge -X theirs caffe/master
'''
After this, open the file at 'caffe-fast-rcnn/include/caffe/layers/python_layer.hpp' and comment out the following line:
'''
.
.
//self_.attr("phase") = static_cast<int>(this->phase_);
self_.attr("setup")(bottom, top);
.
.
'''
Save the file and proceed.

Copy '$PROJECT_ROOT/help/sample_build_configs/caffe-fast-rcnn/Makefile.config' to '$PROJECT_ROOT/dl_algos/py_faster_rcnn/caffe-fast-rcnn/Makefile.config'
Once this is done we can now build caffe-fast-rcnn:
'''
make -j8 && make pycaffe
'''

Go up one folder to 'py_faster_rcnn' and into the 'lib' directory and build the modules there:
'''
cd ../lib            //this directory is $PROJECT_ROOT/dl_algos/py_faster_rcnn/lib
make
'''

To test out if the build was succesful, go to the py_faster_rcnn folder and run the 'demo.py' in the 'tools' folder:
'''
cd ..
./data/scripts/fetch_faster_rcnn_models.sh        //Download the weights
python tools/demo.py
'''
Please note that the weights are downloaded in '$PROJECT_ROOT/dl_algos/py_faster_rcnn/data/faster_rcnn_models'. Once you verify that the build works, I suggest you move this folder outside '$PROJECT_ROOT', to keep the code and models separate. Refer to the new location of this file in the master 'objdet_experiments_conf.ini' file to use it in the object detection framework.

2. darknet (yolo)

Go into the dl_algos/darknet folder:
'''
cd dl_algos/darknet
'''

And build the codebase after editing the Makefile:
'''
//change from 0 to 1
GPU=1
CUDNN=1
OPENCV=1
'''
And then build using:
'''
make
'''

To test if the build was successful:
'''
wget https://pjreddie.com/media/files/yolo.weights        //Download the weights
./darknet detect cfg/yolo.cfg yolo.weights data/dog.jpg
'''
Please note that the weights are downloaded in '$PROJECT_ROOT/dl_algos/darknet'. Once you verify that the build works, I suggest you move this file outside '$PROJECT_ROOT', to keep the code and models separate. Refer to the new location of this file in the master 'objdet_experiments_conf.ini' file to use it in the object detection framework.

**To proceed, have a look at [getting started](GETTING_STARTED.md)**