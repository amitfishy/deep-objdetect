import xml.etree.ElementTree as ET
import pickle
import os
from os.path import join
from dl_algos.py_faster_rcnn.lib.datasets.general_dataset_eval import parse_rec

def convert(size, box):
    #box should be 0-based indices
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def get_classname_list(class_names_file):
    classname_list = []
    with open(class_names_file, 'r') as c_n_f:
        for classname in c_n_f.readlines():
            classname_list.append(classname.strip('\n'))
    return classname_list

def convert_annotation(dataset_type, annotation_source_path, annotation_dest_path, class_names_file, train_mode):
    # if os.path.exists(annotation_dest_path):
    #     return
    annotation_dest_path_file = open(annotation_dest_path, 'w')
    objects = parse_rec(dataset_type, annotation_source_path)

    tree = ET.parse(annotation_source_path)
    size = tree.getroot().find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    classname_list = get_classname_list(class_names_file)


    for obj in objects:
        obj_name = obj['name']
        if obj_name not in classname_list:
            continue
        obj_class_id = classname_list.index(obj_name)
        obj_bbox = obj['bbox']
        obj_sample_mode = obj['sample_mode']
        
        #EXTEND FUNCTIONALITY HERE
        #UPDATE ANNOTATION MODES HERE
        #get all the required annotation info and then filter with dataset_type
        if dataset_type.lower() == 'kitti':
            if obj_sample_mode > train_mode:
                continue
        else:
            if obj_sample_mode != train_mode:
                continue
        
        #b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        obj_b = (float(obj_bbox[0]), float(obj_bbox[2]), float(obj_bbox[1]), float(obj_bbox[3]))
        obj_bb = convert((w,h), obj_b)
        annotation_dest_path_file.write(str(obj_class_id) + " " + " ".join([str(a) for a in obj_bb]) + '\n')

    return

def generate_yolo_format(dataset_type, annotation_directory, image_id_directory, gen_sub_image_ids_directory, image_data_directory, class_names_file, train_mode):
    #ADD OTHER IMAGE EXTENSIONS HERE
    image_exts = ['.jpg','.png','.JPEG']
    yolo_annotation_directory = os.path.join(annotation_directory, 'yolo_format')
    if not os.path.exists(yolo_annotation_directory):
        os.makedirs(yolo_annotation_directory)

    #yolo_image_id_directory is a temporary container which is overwritten every time
    #The original contents are present in gen_sub_image_ids_directory which are copied in yolo format
    yolo_image_id_directory = os.path.join(image_id_directory, 'yolo_format')
    if not os.path.exists(yolo_image_id_directory):
        os.makedirs(yolo_image_id_directory)

    for filename in os.listdir(gen_sub_image_ids_directory):
        source_path = os.path.join(gen_sub_image_ids_directory, filename)
        dest_path = os.path.join(yolo_image_id_directory, filename)

        with open(source_path, 'r') as source_path_file:
            image_inds = [x.strip() for x in source_path_file.readlines()]

        with open(dest_path, 'w') as dest_path_file:
            for image_ind in image_inds:
                for image_ext in image_exts:
                    image_ind_path = os.path.join(image_data_directory, image_ind + image_ext)
                    if os.path.exists(image_ind_path):
                        dest_path_file.write(image_ind_path)
                        dest_path_file.write('\n')
                        annotation_source_path = os.path.join(annotation_directory, image_ind + '.xml')
                        annotation_exists = os.path.exists(annotation_source_path)
                        assert (filename=='test.txt') | annotation_exists, 'xml Annotation for image `{:s}` does not exist: `{:s}`'.format(image_ind_path, annotation_source_path)
                        if annotation_exists:
                            annotation_dest_path = os.path.join(yolo_annotation_directory, image_ind + '.txt')
                            convert_annotation(dataset_type, annotation_source_path, annotation_dest_path, class_names_file, train_mode)
                        break


    return yolo_annotation_directory, yolo_image_id_directory

