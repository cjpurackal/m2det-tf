import cv2
import os
import glob
import numpy as np

class Bbox:

	def __init__(self, xmin, ymin, xmax, ymax, cat = None):
		self.xmin = xmin
		self.xmax = xmax
		self.ymin = ymin
		self.ymax = ymax
		self.cat = cat

	def __getitem__(self,key):
		return getattr(self, key)

def convert_to_bbox(labels_all):
	objs = []
	for label in labels_all:
		label = label.split("\n")
		xmin, ymin, xmax, ymax = label[1].split(" ")
		objs.append(Bbox(int(xmin), int(ymin), int(xmax), int(ymax), int(label[0])))

	return objs

def manip_image_and_label(image_file, objs, config):
	image = cv2.imread(image_file)
	h, w, c = image.shape
	image = cv2.resize(image, (config["IMAGE_H"], config["IMAGE_W"]))
	for obj in objs:

		obj.xmin = int(obj.xmin * float(config['IMAGE_W']) / w)
		obj.xmin = max(min(obj.xmin, config['IMAGE_W']), 0)

		obj.xmax = int(obj.xmax * float(config['IMAGE_W']) / w)
		obj.xmax = max(min(obj.xmax, config['IMAGE_W']), 0)

		obj.ymin = int(obj.ymin * float(config['IMAGE_H']) / h)
		obj.ymin = max(min(obj.ymin, config['IMAGE_H']), 0)

		obj.ymax = int(obj.ymax * float(config['IMAGE_H']) / h)
		obj.ymax = max(min(obj.ymax, config['IMAGE_H']), 0)

	return image, objs

def manip_image(image_file, config):
	image = cv2.imread(image_file)
	image = cv2.resize(image, (config["IMAGE_H"], config["IMAGE_W"]))
	return image

def _interval_overlap(interval_a, interval_b):
	x1, x2 = interval_a
	x3, x4 = interval_b

	if x3 < x1:
		if x4 < x1:
			return 0
		else:
			return min(x2,x4) - x1
	else:
		if x2 < x3:
			 return 0
		else:
			return min(x2,x4) - x3     

def compute_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
	
	intersect = intersect_w * intersect_h

	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	
	union = w1*h1 + w2*h2 - intersect
	
	return float(intersect) / union

def train_test_split(data_path,*args):
    if len(args) ==0:
        percentage_test = 10
    else:
        percentage_test=args[0]
    img_count = len(os.listdir(data_path+"images"))
    file_train = open(data_path+'train.txt', 'w+')  
    file_test = open(data_path+'test.txt', 'w+')
    counter = 1  
    index_test = round(100 / percentage_test) 
    for cats in os.listdir(os.path.join(data_path,"images/")):
        image_path=os.path.join(data_path,"images/",cats)
        for i in glob.iglob(os.path.join(image_path, "*.jpg")):  
            if counter == index_test:
                counter = 1
                file_test.write(i+"\n")
            else:
                file_train.write(i+"\n")
                counter = counter + 1 

def resize(img, labels, img_size):
	img_h, img_w = img.shape[:2]
	ratio = max(img_h, img_w) / img_size
	new_h = int(img_h / ratio)
	new_w = int(img_w / ratio)
	ox = (img_size - new_w) // 2
	oy = (img_size - new_h) // 2
	scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
	out = np.ones((img_size, img_size, 3), dtype=np.uint8) * 127
	out[oy:oy + new_h, ox:ox + new_w, :] = scaled

	scaled_labels = []
	for label in labels:
	    xmin, ymin, xmax, ymax = label[1:5]
	    xmin = (xmin * new_w + ox) / img_size
	    ymin = (ymin * new_h + oy) / img_size
	    xmax = (xmax * new_w + ox) / img_size
	    ymax = (ymax * new_h + oy) / img_size
	    label = [xmin, ymin, xmax, ymax] + label[4:]
	    scaled_labels.append(label)

	return out, scaled_labels