#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import ConfigParser
import os

import dl_algos.py_faster_rcnn.tools.train_net as pfr_train_net
import dl_algos.py_faster_rcnn.tools.test_net as pfr_test_net
from dl_algos.py_faster_rcnn.lib.datasets.general_dataset_eval import general_dataset_eval

from dl_algos.py_faster_rcnn.tools.detect_online import frcnn_online_det
from dl_algos.darknet.python.detect_online import yolo_online_det

import data_utils.split_data as split_data

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#py_faster_rcnn based on the caffe framework by rbgirshick

class misc_params():
	def __init__(self, config):

		self.gpu_id = int(config.get('system_stuff', 'gpu_id'))
		self.image_data_directory = str(config.get('data_dirs', 'image_data_dir'))
		assert os.path.exists(self.image_data_directory), 'Image Data Directory does not exist: `{:s}`'.format(self.image_data_directory)
		self.annotation_directory = str(config.get('data_dirs', 'annotation_dir'))
		assert os.path.exists(self.annotation_directory), 'Annotations Directory does not exist: `{:s}`'.format(self.annotation_directory)
		self.image_id_directory = str(config.get('data_dirs', 'image_id_dir'))
		assert os.path.exists(self.image_id_directory), 'Image IDs Directory does not exist: `{:s}`'.format(self.image_id_directory)
		self.proj_root_directory = str(config.get('data_dirs', 'proj_root_dir'))
		assert os.path.exists(self.proj_root_directory), 'Project Root Directory does not exist: `{:s}`'.format(self.proj_root_directory)

class faster_rcnn_module():
	def __init__(self, config_filename):
		self.config = ConfigParser.ConfigParser()
		self.config.read(config_filename)

		self.use_faster_rcnn = self.config.getboolean('faster_rcnn', 'use_faster_rcnn')
		if self.use_faster_rcnn:
			self._init_general_info()
			self._init_networkspecific_info()

	def _init_general_info(self):
		self.trained_model_path = -1
		self.trained_model_filename = -1

		self.dataset_type = self.config.get('faster_rcnn', 'dataset_type').lower()
		self.experiment_name = self.config.get('faster_rcnn', 'experiment_name').lower()
		
		self.train = self.config.getboolean('faster_rcnn', 'train')
		self.train_split = self.config.get('faster_rcnn', 'train_split')
		self.train_mode = int(self.config.get('faster_rcnn', 'train_mode'))

		self.test = self.config.getboolean('faster_rcnn', 'test')
		self.test_split = self.config.get('faster_rcnn', 'test_split')
		self.test_mode = int(self.config.get('faster_rcnn', 'test_mode'))

		self.evaluate = self.config.getboolean('faster_rcnn', 'evaluate')
		self.use07metric = self.config.getboolean('faster_rcnn', 'use07metric')
		
		self.output_directory = self.config.get('faster_rcnn', 'output_directory')
		if not os.path.exists(self.output_directory):
			os.makedirs(self.output_directory)
		
		self.num_iterations = int(self.config.get('faster_rcnn', 'num_iterations'))

		self.use_validation_experiments = self.config.getboolean('faster_rcnn', 'use_validation_experiments')
		self.train_data_fraction = float(self.config.get('faster_rcnn', 'train_data_fraction'))
		self.num_repetitions = 1

		if self.use_validation_experiments:
			self.num_repetitions = int(self.config.get('faster_rcnn', 'num_repetitions'))

		self.network_name = str(self.config.get('faster_rcnn', 'network_name')).lower()
		self.output_model_prefix = str(self.config.get('faster_rcnn', 'output_model_prefix')).lower()

		self.use_pretrained_weights = self.config.getboolean('faster_rcnn', 'use_pretrained_weights')
		if self.use_pretrained_weights:
			self.pretrained_weights_file = str(self.config.get('faster_rcnn', 'pretrained_weights_file'))
			assert os.path.isfile(self.pretrained_weights_file), 'Pre-trained weights file does not exist: `{:s}`'.format(self.pretrained_weights_file)
			self.trained_model_filename = self._get_next_model_name()
		else:
			self.pretrained_weights_file = -1
			self.trained_model_filename = self.output_model_prefix + '_' + str(self.num_iterations) + '.caffemodel'

		self.use_trained_weights_test = self.config.getboolean('faster_rcnn', 'use_trained_weights_test')
		if not self.use_trained_weights_test:
			self.weights_file_test = str(self.config.get('faster_rcnn', 'weights_file_test'))
			assert os.path.isfile(self.weights_file_test), 'Weights file for testing does not exist: `{:s}`'.format(self.weights_file_test)
		else:
			assert self.train, '`train` is set to False, enable `train`, or set `use_trained_weights_test` to False and add a weights file for testing'
			self.weights_file_test = -1

		return

	def _init_networkspecific_info(self):
		self.misc = misc_params(self.config)
		self.code_directory = os.path.join(self.misc.proj_root_directory, 'dl_algos', 'py_faster_rcnn')
		self.class_names_file = self.config.get('faster_rcnn', 'class_names_file')
		assert os.path.isfile(self.class_names_file), 'Class Names file does not exist: `{:s}`'.format(self.class_names_file)

		if self.use_faster_rcnn:
			self.solver_proto_file = str(self.config.get('faster_rcnn', 'solver_proto_file'))
			assert os.path.isfile(self.solver_proto_file) or (not self.train), 'Solver Prototxt file does not exist: `{:s}`'.format(self.solver_proto_file)
			self.train_proto_file = str(self.config.get('faster_rcnn', 'train_proto_file'))
			assert os.path.isfile(self.train_proto_file) or (not self.train), 'Train Prototxt file does not exist: `{:s}`'.format(self.train_proto_file)
		if self.use_faster_rcnn:
			self.test_proto_file = str(self.config.get('faster_rcnn', 'test_proto_file'))
			assert os.path.isfile(self.test_proto_file) or (not self.test), 'Test Prototxt file does not exist: `{:s}`'.format(self.test_proto_file)
		else:
			self.test_proto_file = str(self.config.get('faster_rcnn', 'test_proto_file'))
			assert os.path.isfile(self.test_proto_file), 'Test Prototxt file does not exist: `{:s}`'.format(self.test_proto_file)

		self.cfg_file = self.config.get('faster_rcnn', 'cfg_file')
		assert os.path.isfile(self.cfg_file), 'Cfg File for Faster RCNN does not exist: `{:s}`'.format(self.cfg_file)

		return

	def _init_online_info(self):
		self.weights_file_online = str(self.config.get('faster_rcnn', 'weights_file_online'))
		assert os.path.isfile(self.weights_file_online), 'Online weights file does not exist: `{:s}`'.format(self.weights_file_online)

		self.detection_thresh_online = float(self.config.get('faster_rcnn', 'detection_thresh_online'))
		assert (self.detection_thresh_online >= 0) and (self.detection_thresh_online <= 1), 'Detection Threshold should lie between 0 and 1. Value given: `{:f}`'.format(self.detection_thresh_online)
			
		self.nms_thresh_online = float(self.config.get('faster_rcnn', 'nms_thresh_online'))
		assert (self.nms_thresh_online >= 0) and (self.nms_thresh_online <= 1), 'Non-Maximum Suppression Threshold should lie between 0 and 1. Value given: `{:f}`'.format(self.nms_thresh_online)

		return

	def _get_next_model_name(self):
		#base file name
		name = os.path.basename(self.pretrained_weights_file)
		#base file name without extension
		name = os.path.splitext(name)[0]
		current_iters = int(name.split('_')[-1])

		next_iters = self.num_iterations + current_iters
		new_filename = self.output_model_prefix + '_' + str(next_iters) + '.caffemodel'
		return new_filename

	def faster_rcnn_interface(self):
		if self.use_faster_rcnn:
			if self.use_validation_experiments:
				assert (self.train and self.test), 'If use validation_experiments=True, keep train and test: True'
				assert (self.train_data_fraction > 0) and (self.train_data_fraction < 1.0), 'If use_validation_experiments=True, keep 0 < train_data_fraction < 1.0'
				assert self.test_split == 'val', 'If use_validation_experiments=True, keep test_split: val'
				assert self.evaluate, 'If use_validation_experiments=True, keep evaluate: True'
				assert self.use_trained_weights_test, 'If use_validation_experiments=True, keep use_trained_weights_test: True'

				output_results_folder = os.path.join(self.output_directory, self.experiment_name, 'results', 'temp')
				if not os.path.exists(output_results_folder):
					os.makedirs(output_results_folder) 
				output_file = os.path.join(output_results_folder, 'AveragePrecision.txt')
				with open(output_file, 'w') as file:
					file.write('Average Precisions for each Split:\n')

				for num_iter in xrange(self.num_repetitions):
					#get random data from original full trainval and split it based on self.train_data_fraction
					split_data.get_split_data(self.misc.image_id_directory, self.train_data_fraction, self.use_validation_experiments)
					self.faster_rcnn_train()
					mAP, aps = self.faster_rcnn_test()

					#write mAP, aps into output_file (append)
					with open(output_file, 'a') as file:
						file.write('Split #{:s}:\n'.format(str(num_iter+1)))
						file.write("Average Precisions For each class:\n");
						for ap in aps:
							file.write('{:s} '.format(str(ap)))
						file.write('\n')
						file.write('Mean Average Precision:\n')
						file.write('{:s}\n'.format(str(mAP)))                    
			else:
				assert (self.train_data_fraction <= 1.0) and (self.train_data_fraction > 0), 'Use train_data_fraction such that: 0 < train_data_fraction <= 1.0'
				#get random data from original full trainval and split it based on self.train_data_fraction
				split_data.get_split_data(self.misc.image_id_directory, self.train_data_fraction, self.use_validation_experiments)
				if self.train:
					self.faster_rcnn_train()
				if self.test:
					self.faster_rcnn_test()

		return

	def faster_rcnn_train(self):
		self.trained_model_path = str(pfr_train_net.train_controller(self))
		return

	def faster_rcnn_test(self):
		mAP, aps = pfr_test_net.test_controller(self)

		if self.evaluate:
			if self.use_validation_experiments:
				return mAP, aps
			Split = str(int(self.train_data_fraction*100)) + '-' + str(int(100-self.train_data_fraction*100))
			output_results_folder = os.path.join(self.output_directory, self.experiment_name, 'results', Split)
			output_file = os.path.join(output_results_folder, 'AveragePrecision.txt')
			with open(output_file, 'w') as file:
				file.write("Average Precisions For each class:\n");
				for ap in aps:
					file.write('{:s} '.format(str(ap)))
				file.write('\n')
				file.write('Mean Average Precision:\n')
				file.write('{:s}\n'.format(str(mAP)))
		return

	def faster_rcnn_online_init(self):
		self._init_networkspecific_info()
		self._init_online_info()
		self.ol_detector = frcnn_online_det(self)
		return

	def faster_rcnn_online(self, im):
		all_dets = self.ol_detector.det(im)
		#print 'frcnn: '
		#print all_dets
		#print '----------'
		return all_dets


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# yolo module based of darknet framework. runs c code using terminal commands through os.system()

class yolo_module():
	def __init__(self, config_filename):
		self.config = ConfigParser.ConfigParser()
		self.config.read(config_filename)        
		
		self.use_yolo = self.config.getboolean('yolo', 'use_yolo')

		if self.use_yolo:
			self._init_general_info()
			self._init_networkspecific_info()

	def _init_general_info(self):
		self._yolo_annotation_directory = -1
		self._yolo_image_id_directory = -1

		self.trained_model_path = -1
		self.trained_model_filename = -1
		
		self.dataset_type = self.config.get('yolo', 'dataset_type').lower()
		self.experiment_name = self.config.get('yolo', 'experiment_name').lower()
		
		self.train = self.config.getboolean('yolo', 'train')
		self.train_split = self.config.get('yolo', 'train_split')
		self.train_mode = int(self.config.get('yolo', 'train_mode'))

		self.test = self.config.getboolean('yolo', 'test')
		self.test_split = self.config.get('yolo', 'test_split')
		self.test_mode = int(self.config.get('yolo', 'test_mode'))

		self.evaluate = self.config.getboolean('yolo', 'evaluate')
		self.use07metric = self.config.getboolean('yolo', 'use07metric')

		self.output_directory = self.config.get('yolo', 'output_directory')
		if not os.path.exists(self.output_directory):
			os.makedirs(self.output_directory)
		
		self.num_iterations = int(self.config.get('yolo', 'num_iterations'))

		self.use_validation_experiments = self.config.getboolean('yolo', 'use_validation_experiments')
		self.train_data_fraction = float(self.config.get('yolo', 'train_data_fraction'))
		self.num_repetitions = 1
		
		if self.use_validation_experiments:
			self.num_repetitions = int(self.config.get('yolo', 'num_repetitions'))

		self.network_name = str(self.config.get('yolo', 'network_name')).lower()
		self.output_model_prefix = str(self.config.get('yolo', 'output_model_prefix')).lower()

		self.use_pretrained_weights = self.config.getboolean('yolo', 'use_pretrained_weights')
		if self.use_pretrained_weights:
			self.pretrained_weights_file = str(self.config.get('yolo', 'pretrained_weights_file'))
			assert os.path.isfile(self.pretrained_weights_file), 'Pre-trained weights file does not exist: `{:s}`'.format(self.pretrained_weights_file)
			self.trained_model_filename = self._get_next_model_name()
		else:
			self.pretrained_weights_file = -1
			self.trained_model_filename = self.output_model_prefix + '_' + str(self.num_iterations) + '.weights'

		self.use_trained_weights_test = self.config.getboolean('yolo', 'use_trained_weights_test')
		if not self.use_trained_weights_test:
			self.weights_file_test = str(self.config.get('yolo', 'weights_file_test'))
			assert os.path.isfile(self.weights_file_test), 'Weights file for testing does not exist: `{:s}`'.format(self.weights_file_test)
		else:
			assert self.train, '`train` is set to False, enable `train`, or set `use_trained_weights_test` to False and add a weights file for testing'
			self.weights_file_test = -1

		return

	def _init_networkspecific_info(self):
		self.misc = misc_params(self.config)
		self.code_directory = os.path.join(self.misc.proj_root_directory, 'dl_algos', 'darknet')
		self.class_names_file = self.config.get('yolo', 'class_names_file')
		assert os.path.isfile(self.class_names_file), 'Class Names file does not exist: `{:s}`'.format(self.class_names_file)

		self.network_cfg_file = str(self.config.get('yolo', 'network_cfg_file'))
		assert os.path.isfile(self.network_cfg_file), 'Network config File for yolo does not exist: `{:s}`'.format(self.network_cfg_file)

		return

	def _init_online_info(self):
		self.weights_file_online = str(self.config.get('yolo', 'weights_file_online'))
		assert os.path.isfile(self.weights_file_online), 'Online weights file does not exist: `{:s}`'.format(self.weights_file_online)

		self.detection_thresh_online = float(self.config.get('yolo', 'detection_thresh_online'))
		assert (self.detection_thresh_online >= 0) and (self.detection_thresh_online <= 1), 'Detection Threshold should lie between 0 and 1. Value given: `{:f}`'.format(self.detection_thresh_online)
			
		self.nms_thresh_online = float(self.config.get('yolo', 'nms_thresh_online'))
		assert (self.nms_thresh_online >= 0) and (self.nms_thresh_online <= 1), 'Non-Maximum Suppression Threshold should lie between 0 and 1. Value given: `{:f}`'.format(self.nms_thresh_online)

		return

	def _get_next_model_name(self):
		#base file name
		name = os.path.basename(self.pretrained_weights_file)
		#base file name without extension
		name = os.path.splitext(name)[0]
		current_iters = int(name.split('_')[-1])

		next_iters = self.num_iterations + current_iters
		new_filename = self.output_model_prefix + '_' + str(next_iters) + '.weights'
		return new_filename

	def yolo_interface(self):
		if self.use_yolo:
			if self.use_validation_experiments:
				assert (self.train and self.test), 'If use validation_experiments=True, keep train and test: True'
				assert (self.train_data_fraction > 0) and (self.train_data_fraction < 1.0), 'If use_validation_experiments=True, keep 0 < train_data_fraction < 1.0'
				assert self.test_split == 'val', 'If use_validation_experiments=True, keep test_split: val'
				assert self.evaluate, 'If use_validation_experiments=True, keep evaluate: True'
				assert self.use_trained_weights_test, 'If use_validation_experiments=True, keep use_trained_weights_test: True'

				output_results_folder = os.path.join(self.output_directory, self.experiment_name, 'results', 'temp')
				if not os.path.exists(output_results_folder):
					os.makedirs(output_results_folder) 
				output_file = os.path.join(output_results_folder, 'AveragePrecision.txt')
				with open(output_file, 'w') as file:
					file.write('Average Precisions for each Split:\n')

				for num_iter in xrange(self.num_repetitions):
					#get random data from original full trainval and split it based on self.train_data_fraction
					self._yolo_annotation_directory, self._yolo_image_id_directory = split_data.generate_yolo_format_data(self.dataset_type, self.misc.annotation_directory, self.misc.image_id_directory, self.train_data_fraction, self.use_validation_experiments, self.misc.image_data_directory, self.class_names_file, self.train_mode)
					self.yolo_train()
					mAP, aps = self.yolo_test()
					#write mAP, aps into output_file (append)
					with open(output_file, 'a') as file:
						file.write('Split #{:s}:\n'.format(str(num_iter+1)))
						file.write("Average Precisions For each class:\n");
						for ap in aps:
							file.write('{:s} '.format(str(ap)))
						file.write('\n')
						file.write('Mean Average Precision:\n')
						file.write('{:s}\n'.format(str(mAP)))
			else:
				assert (self.train_data_fraction <= 1.0) and (self.train_data_fraction > 0), 'Use train_data_fraction such that: 0 < train_data_fraction <= 1.0'
				#get random data from original full trainval and split it based on self.train_data_fraction
				self._yolo_annotation_directory, self._yolo_image_id_directory = split_data.generate_yolo_format_data(self.dataset_type, self.misc.annotation_directory, self.misc.image_id_directory, self.train_data_fraction, self.use_validation_experiments, self.misc.image_data_directory, self.class_names_file, self.train_mode)
				if self.train:
					self.yolo_train()
				if self.test:
					self.yolo_test()

		return

	def yolo_train(self):
		#get current directory
		cwd = os.getcwd()
		#change directory to code folder
		os.chdir(self.code_directory)

		code_exec_sys = './darknet detector'
		gpu_sys = '-gpus ' + str(self.misc.gpu_id)
		net_cfg_sys = '-net_config_file ' + self.network_cfg_file
		class_names_sys = '-class_names_file ' + self.class_names_file
		annotations_sys = '-annotation_folder ' + self._yolo_annotation_directory
		
		if self.use_validation_experiments:
			subfolder = 'temp'
		else:
			subfolder = str(int(self.train_data_fraction*100)) + '-' + str(int((100-self.train_data_fraction*100)))
		output_model_folder = os.path.join(self.output_directory, self.experiment_name, 'models', subfolder)
		if not os.path.exists(output_model_folder):
			os.makedirs(output_model_folder) 

		output_model_sys = '-output_model_folder ' + output_model_folder

		if self.use_pretrained_weights:
			weights_sys = '-weights_file ' + self.pretrained_weights_file + ' '
		else:
			weights_sys = ''

		#yolo_train is called only when self.train == 1
		train_sys = '-train 1'
		train_sets_sys = '-train_sets_file ' + os.path.join(self._yolo_image_id_directory, self.train_split + '.txt')

		#max_batches and num_iterations are the same thing
		max_batches_iter_sys = '-max_batches ' + str(self.num_iterations)

		output_model_name_sys = '-output_model_filename ' + self.trained_model_filename

		#run c code - train using yolo
		cmd = code_exec_sys + ' ' + gpu_sys + ' ' + net_cfg_sys + ' ' + class_names_sys + ' ' + annotations_sys + ' ' + output_model_sys + ' ' + weights_sys + train_sys + ' ' + train_sets_sys + ' ' + max_batches_iter_sys + ' ' + output_model_name_sys
		print 'Running: {:s}'.format(cmd)
		trainres = os.system(cmd)
		assert not trainres, 'ERROR: SOMETHING WENT WRONG WHILE TRYING TO TRAIN DATA IN DARKNET. REVIEW THE CONFIG FILES INVOLVED.'

		self.trained_model_path = os.path.join(output_model_folder, self.trained_model_filename)

		return

	def yolo_test(self):
		#get current directory
		cwd = os.getcwd()
		#change directory to code folder
		os.chdir(self.code_directory)

		code_exec_sys = './darknet detector'
		gpu_sys = '-gpus ' + str(self.misc.gpu_id)
		net_cfg_sys = '-net_config_file ' + self.network_cfg_file
		class_names_sys = '-class_names_file ' + self.class_names_file

		if self.use_validation_experiments:
			subfolder = 'temp'
		else:
			subfolder = str(int(self.train_data_fraction*100)) + '-' + str(int((100-self.train_data_fraction*100)))
		output_results_folder = os.path.join(self.output_directory, self.experiment_name, 'results', subfolder)
		if not os.path.exists(output_results_folder):
			os.makedirs(output_results_folder) 
		
		results_sys = '-output_results_folder ' + output_results_folder

		if not self.use_trained_weights_test:
			weights_sys = '-weights_file ' + self.weights_file_test
		else:
			weights_sys = '-weights_file ' + self.trained_model_path

		#yolo_test is called only when self.test == 1
		test_sys = '-test 1'
		test_sets_sys = '-test_sets_file ' + os.path.join(self._yolo_image_id_directory, self.test_split + '.txt')

		#run c code - test using yolo - get detection outputs (no mAP)
		cmd = code_exec_sys + ' ' + gpu_sys + ' ' + net_cfg_sys + ' ' + class_names_sys + ' ' + results_sys + ' ' + weights_sys + ' ' + test_sys + ' ' + test_sets_sys
		print 'Running: {:s}'.format(cmd)
		testres = os.system(cmd)
		assert not testres, 'ERROR: SOMETHING WENT WRONG WHILE TRYING TO TEST DATA IN DARKNET. REVIEW THE CONFIG FILES INVOLVED.'

		#Find mAP using the same evaluation script as faster-rcnn
		if self.evaluate:
			print 'Finding mAP over output detections using faster rcnn evaluation script'
			mAP, aps = self.evaluate_yolo_frcnn_script(output_results_folder)
			if self.use_validation_experiments:
				return mAP, aps

			output_file = os.path.join(output_results_folder, 'AveragePrecision.txt')
			with open(output_file, 'w') as file:
				file.write("Average Precisions For each class:\n");
				for ap in aps:
					file.write('{:s} '.format(str(ap)))
				file.write('\n')
				file.write('Mean Average Precision:\n')
				file.write('{:s}\n'.format(str(mAP)))
		return

	def evaluate_yolo_frcnn_script(self, output_results_folder):
		classname_list = []
		with open(self.class_names_file, 'r') as c_n_f:
			for classname in c_n_f.readlines():
				classname_list.append(classname.strip())
		if self.use_validation_experiments:
			imagesetfile = os.path.join(self.misc.image_id_directory, 'temp', self.test_split + '.txt')
		else:
			Split = str(int(self.train_data_fraction*100)) + '-' + str(int(100-self.train_data_fraction*100))
			imagesetfile = os.path.join(self.misc.image_id_directory, Split, self.test_split + '.txt')

		annopath = os.path.join(self.misc.annotation_directory, '{:s}.xml')

		aps = []
		# The PASCAL VOC metric changed in 2010 to area of AP curve
		print 'VOC07 metric? ' + ('Yes' if self.use07metric else 'No')
		for i, cls_name in enumerate(classname_list):
			if (cls_name == '__background__'):
				continue
			print 'Evaluating Class: `{:s}`'.format(cls_name)
			filename = os.path.join(output_results_folder, 'comp4_det_{:s}_{:s}.txt'.format(self.test_split, cls_name))
			rec, prec, ap = general_dataset_eval(self.dataset_type, filename, annopath, imagesetfile, cls_name, self.test_mode, ovthresh=0.5, use_07_metric=self.use07metric)
			aps += [ap]
			print('AP for {} = {:.4f}'.format(cls_name, ap))

		mAP = np.mean((aps))
		print('Mean AP = {:.4f}'.format(mAP))
		print('~~~~~~~~')
		print('Yolo Results:')
		for ap in aps:
			print('{:.3f}'.format(ap))
		print('{:.3f}'.format(mAP))
		print('~~~~~~~~')
		print('')
		print('--------------------------------------------------------------')
		print('Results computed with the **unofficial** Python eval code.')
		print('Results should be very close to the official MATLAB eval code for VOC')
		print('--------------------------------------------------------------')

		return mAP, aps
	
	def yolo_online_init(self):
		self._init_networkspecific_info()
		self._init_online_info()
		self.ol_detector = yolo_online_det(self)

		return

	def yolo_online(self, im):
		all_dets = self.ol_detector.det(im)
		#print 'yolo: '
		#print all_dets
		#print '----------'
		return all_dets

#Run this script directly to do the object detection experiments. Import this as a module to use the online detector.
if __name__ == '__main__':
	rcnnexp = faster_rcnn_module('objdet_experiments_conf.ini')
	rcnnexp.faster_rcnn_interface()

	yoloexp = yolo_module('objdet_experiments_conf.ini')
	yoloexp.yolo_interface()