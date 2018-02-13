import objdet_experiments as objexp
import cv2

def displayDet(image, all_dets):
	det_image = image

	for det in all_dets:
		name = det[0]
		score = det[1]
		bbox = det[2]
		cv2.rectangle(det_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,255,0),2)
		cv2.putText(det_image, name, (int(bbox[0]),int(bbox[1])), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)
		print 'Class Name: {:s}, Score: {:f}, Bounding Box: [x1(left): {:d}, y1(top): {:d}, x2(right): {:d}, y2(bottom): {:d} ]'.format(name, score, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
	#cv2.rectangle(det_image, ())
	cv2.imshow('display', det_image)
	cv2.waitKey(0)
	return

def process_image_pfrcnn(rcnnexp, imagefile):
	image = cv2.imread(imagefile)
	all_dets = rcnnexp.faster_rcnn_online(image)
	print 'Faster - RCNN detection'
	displayDet(image, all_dets)
	print '-----------------------'
	return

def process_image_yolo(yoloexp, imagefile):
	image = cv2.imread(imagefile)
	all_dets = yoloexp.yolo_online(image)
	print 'yolo detection'
	displayDet(image, all_dets)
	print '-----------------------'
	return

#The model to be trained should be defined in this config file. Look at $PROJECT_ROOT/help/docs/BUILD_INSTRUCTIONS.md for how to download some pre-trained models.
#Be sure to put the appropriate prototxt, cfg and class names files with the corresponding py-faster-rcnn weights model in the master objdet_experiments_conf.ini file
#Be sure to put the appropriate cfg and class names files with the corresponding darknet weights model in the master objdet_experiments_conf.ini file
#The sample config files for the models downloaded from $PROJECT_ROOT/help/docs/BUILD_INSTRUCTIONS.md are given in the $PROJECT_ROOT/config_data directory
config = 'objdet_experiments_conf.ini'

rcnnexp = objexp.faster_rcnn_module(config)
rcnnexp.faster_rcnn_online_init()

yoloexp = objexp.yolo_module('objdet_experiments_conf.ini')
yoloexp.yolo_online_init()

#Enter some image file paths (absolute) as a list here
#This is where I stored images from the PASCAL set
imagefiles=['/home/amitsinha/data/frameworktest/VOCdevkit/VOC2012/JPEGImages/2008_000145.jpg',
			'/home/amitsinha/data/frameworktest/VOCdevkit/VOC2012/JPEGImages/2008_000176.jpg',
			'/home/amitsinha/data/frameworktest/VOCdevkit/VOC2012/JPEGImages/2008_000185.jpg',
			'/home/amitsinha/data/frameworktest/VOCdevkit/VOC2012/JPEGImages/2008_000190.jpg',
			'/home/amitsinha/data/frameworktest/VOCdevkit/VOC2012/JPEGImages/2008_000192.jpg',
			'/home/amitsinha/data/frameworktest/VOCdevkit/VOC2012/JPEGImages/2008_000199.jpg']
for imagef in imagefiles:
	process_image_pfrcnn(rcnnexp, imagef)
	process_image_yolo(yoloexp, imagef)
