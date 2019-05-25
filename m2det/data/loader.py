import numpy as np
import os
import glob
import cv2
from m2det.data.utils import resize
from m2det.utils import generate_anchors
from m2det.data.label_transformer import transformer1

class Loader:

	def __init__(self, config):
		images_dir = config["train"]["images_dir"]
		labels_dir = config["train"]["labels_dir"]
		assert os.path.exists(images_dir), "%s doesn't exis"%images_dir
		assert os.path.exists(labels_dir), "%s doesn't exis"%labels_dir
		self.images = [i for i in glob.glob("%s/*.jpg"%images_dir)]
		self.labels = [i.replace("jpg", "txt") for i in glob.glob("%s/*.jpg"%images_dir)]
		assert len(self.images)==len(self.labels), "Different Number of Images and Labels found!"
		self.batch_size = config["train"]["batch_size"]
		self.batch_ptr = 0
		self.num_classes = config["model"]["classes"]
		self.input_size = config["model"]["input_size"]
		self.iou_thresh = config["anchors"]["iou_thresh"]
		self.anchors = generate_anchors()

	def next_batch(self, ptr=None):
		x_batch = []
		y_batch = []
		head = self.batch_ptr
		tail = self.batch_ptr+self.batch_size
		for image, label in zip(self.images[head:tail], self.labels[head:tail]):
			img = cv2.imread(image)
			boxes = []
			with open(label) as lf:
				for line in lf.readlines():
					ix, x1, y1, x2, y2 = line.split("\t")
					one_hot_ix = np.eye(self.num_classes)[int(ix)]
					boxes.append([float(x1), float(y1), float(x2), float(y2)]+one_hot_ix.tolist())			
			#resize images to 320 x 320 and correct labels accordingly
			img, boxes = resize(img, boxes, self.input_size)
			#process boxes and return the truth tensor
			boxes = np.array(boxes)
			labels = transformer1(boxes, self.num_classes, self.iou_thresh)
			x_batch.append(img)
			y_batch.append(labels)

		return np.array(x_batch), np.array(y_batch)

	def set_batch_ptr(self, batch_ptr):
		self.batch_ptr = batch_ptr
		