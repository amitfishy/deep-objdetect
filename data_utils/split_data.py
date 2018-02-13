import os
from random import shuffle

import pascalvoc_to_yolo

def generate_yolo_format_data(dataset_type, annotation_directory, image_id_directory, train_data_fraction, use_validation_experiments, image_data_directory, class_names_file, train_mode):
	get_split_data(image_id_directory, train_data_fraction, use_validation_experiments)

	if use_validation_experiments:
		gen_sub_image_ids_directory = os.path.join(image_id_directory, 'temp')
	else:
		Split = str(int(train_data_fraction*100)) + '-' + str(int((100-train_data_fraction*100)))
		gen_sub_image_ids_directory = os.path.join(image_id_directory, Split)

	yolo_annotation_directory, yolo_image_id_directory = pascalvoc_to_yolo.generate_yolo_format(dataset_type, annotation_directory, image_id_directory, gen_sub_image_ids_directory, image_data_directory, class_names_file, train_mode)

	return yolo_annotation_directory, yolo_image_id_directory

def generate_split_data(train_image_id_path, train_data_fraction, out_train_image_id_path, out_val_image_id_path):
	ImageIDs = []
	with open(train_image_id_path, 'r') as train_image_id_file:
		for line in train_image_id_file.readlines():
			ImageIDs.append(line.strip())
	shuffle(ImageIDs)

	with open(out_train_image_id_path, 'w') as out_train_image_id_file:
		for ImageID in sorted(ImageIDs[0:int(train_data_fraction*len(ImageIDs))]):
			out_train_image_id_file.write(ImageID)
			out_train_image_id_file.write('\n')

	with open(out_val_image_id_path, 'w')  as out_val_image_id_file:
		for ImageID in sorted(ImageIDs[int(train_data_fraction*len(ImageIDs)):]):
			out_val_image_id_file.write(ImageID)
			out_val_image_id_file.write('\n')
	return


def generate_temp_split_data(image_id_directory, train_data_fraction):
	temp_split_directory = os.path.join(image_id_directory, 'temp')
	if not os.path.exists(temp_split_directory):
		os.makedirs(temp_split_directory)

	train_image_id_path = os.path.join(image_id_directory, 'train.txt')

	out_train_image_id_path = os.path.join(temp_split_directory, 'train.txt')
	out_val_image_id_path = os.path.join(temp_split_directory, 'val.txt')
	
	#divide train.txt > train and val
	generate_split_data(train_image_id_path, train_data_fraction, out_train_image_id_path, out_val_image_id_path)
	return

def generate_fixed_split_data(image_id_directory, train_data_fraction):
	Split = str(int(train_data_fraction*100)) + '-' + str(int((100-train_data_fraction*100)))
	fixed_split_directory = os.path.join(image_id_directory, Split)
	if not os.path.exists(fixed_split_directory):
		os.makedirs(fixed_split_directory)

	train_image_id_path = os.path.join(image_id_directory, 'train.txt')
	test_image_id_path = os.path.join(image_id_directory, 'test.txt')

	out_train_image_id_path = os.path.join(fixed_split_directory, 'train.txt')
	out_val_image_id_path = os.path.join(fixed_split_directory, 'val.txt')
	out_test_image_id_path = os.path.join(fixed_split_directory, 'test.txt')

	if not (os.path.exists(out_train_image_id_path) and os.path.exists(out_val_image_id_path)):
		#divide train.txt > train and val
		generate_split_data(train_image_id_path, train_data_fraction, out_train_image_id_path, out_val_image_id_path)
		os.system('cp ' + test_image_id_path + ' ' + out_test_image_id_path)
	return

def get_split_data(image_id_directory, train_data_fraction, use_validation_experiments):
	if use_validation_experiments:
		generate_temp_split_data(image_id_directory, train_data_fraction)
	else:
		generate_fixed_split_data(image_id_directory, train_data_fraction)
	return