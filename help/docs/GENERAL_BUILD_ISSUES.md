# **Some Common Issues during the Build and Runtime**

## For py_faster_rcnn

### 1. cannot find #include "caffe/proto/caffe.pb.h" - [Issue #1761](https://github.com/BVLC/caffe/issues/1761)

Enter the py_faster_rcnn directory and use the following:
```
protoc src/caffe/proto/caffe.proto --cpp_out=.
mkdir include/caffe/proto
mv src/caffe/proto/caffe.pb.h include/caffe/proto
```

### 2. Errors relating to cudnn and cuda version. - [Issue #237](https://github.com/rbgirshick/py-faster-rcnn/issues/237)

Enter the py_faster_rcnn directory and use the following:
```
cd caffe-fast-rcnn  
git remote add caffe https://github.com/amitfishy/caffe
git fetch caffe
git merge -X theirs caffe/master
```
After this, open the file at `caffe-fast-rcnn/include/caffe/layers/python_layer.hpp` and comment out the following line:
```
.
.
//self_.attr("phase") = static_cast<int>(this->phase_);
self_.attr("setup")(bottom, top);
.
.
```
Save the file and proceed.

### 3. Erros relating to improperly set environment variables:

Some errors may arise due to improperly set environment variables relating to cuda, cudnn, python, nvidia drivers

- runtime linking issues with dynamic libraries
```
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

- missing -lcudnn (static)
```
export LIBRARY_PATH=/usr/local/cuda-9.0/targets/x86_64-linux/lib:$LIBRARY_PATH
```

- missing cuda or cudnn header files
```
export CPATH=/usr/local/cuda-9.0/targets/x86_64-linux/include:$CPATH
```

- missing pyconfig.h
```
export CPLUS_INCLUDE_PATH=/usr/include/python2.7:$CPLUS_INCLUDE_PATH
```

- A few general env variables for use with Nvidia GPUs in general
```
export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/nvidia-384:$LD_LIBRARY_PATH
```