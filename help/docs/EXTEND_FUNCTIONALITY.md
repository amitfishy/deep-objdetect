# **EXTEND FUNCTIONALITY**

For most users, if you just want to use the entire set of annotations and skip any additional filtering then skip this section.

The parameter `dataset_type` can be extended further by modifying the source code. As of now it supports only the `kitti` type which involves filtering data into `easy`, `moderate` and `hard` categories as defined [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).

To add additional parameters in the annotations and different filtering levels for your own custom `dataset_type` modify the source code in the following places:

The places where the source code should be modified are marked with the comments:
```
    #EXTEND FUNCTIONALITY HERE
    #UPDATE ANNOTATION MODES HERE
```
Followed by a short description of the context it needs to updated in.

1. function name: `parse_rec()` in `$PROJECT_ROOT/dl_algos/py_faster_rcnn/lib/datasets/general_dataset_eval.py`
This function allows you to add additional annotation data fields to store for a particular image. Add an `else if` block and check for your custom `dataset_type` here and add on the additional annotation entries to the structure variable.

2. function name: `determine_sample_mode()` in `$PROJECT_ROOT/dl_algos/py_faster_rcnn/lib/datasets/general_dataset_eval.py`
This function allows you to filter data for training and testing and refers to the `train_mode` and `test_mode` in the master `objdet_experiments_conf.ini` file. To extend this, add an `else if` block and check for your custom `dataset_type` here and assign a `sample_mode` (the level) depending on the annotation fields (refer to the `kitti` part of the code for an example). It is generally more meaningful to make sure that lower levels are subsets of the higher levels. Assign a lower number (`sample_mode` field) to the lower levels and a higher number to higher levels (Note that this is not compulsory but recommended to make it easier to understand).

3. function name: `general_dataset_eval()` in `$PROJECT_ROOT/dl_algos/py_faster_rcnn/lib/datasets/general_dataset_eval.py`
If evaluation needs to be carried out on additional `dataset_type`, modify this function.
There are 2 parts here:
- `ValidInds` : Filter out higher level sample modes and get a boolean array for evaluation purposes
- `MinHeight` : You can set the minimum height of bounding boxes here (for evaluation)

4. function name: `load_general_dataset_annotation()` in `$PROJECT_ROOT/dl_algos/py_faster_rcnn/lib/datasets/general_dataset.py`
For adding additional annotation data for a different `dataset_type` and for filtering data for training networks with faster-rcnn modify the source code here, keeping in mind the conditions you set earlier on the `sample_mode` (in `parse_rec()` and `determine_sample_mode()`). Keep in mind that you will have to add all annotation fields again even if you have added them in `parse_rec()`.

5. function name: `convert_annotation()` in `$PROJECT_ROOT/data_utils/pascal_to_yolo.py`
For filtering data for training networks in darknet (yolo/yolov2) modify the source code here, keeping in mind the conditions you set earlier on the `sample_mode` (in `parse_rec()` and `determine_sample_mode()`).