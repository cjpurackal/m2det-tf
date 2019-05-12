import numpy as np
import tensorflow as tf


def bbox_to_xml(box):
	'''
	args:
		box: (N, (center_x, center_y, width, height))
	
	return:
		box : (N, x1, y1, x2, y2)
			(top-left, bottom-right)
	'''
	# x1  		x2         
	#(top-left)(botm-right)= (center_x)-(width)*0.5   ,   (center_x) + (width) * 0.5
	box[:, 0], box[:, 2] = (box[:, 0] - box[:, 2] * 0.5), (box[:, 0] + box[:, 2] * 0.5)
	# y1       y2
	#(top-left)(botm-right)= (center_y)+(height) * 0.5,   (center_y) - (height) * 0.5
	box[:, 1], box[:, 3] = (box[:, 1] + box[:, 3] * 0.5), (box[:, 1] - box[:, 3] * 0.5)

	return box


def compute_overlaps(boxes1, boxes2):
	'''Computes IoU overlaps between two sets of boxes.
	boxes1, boxes2: [N, (y1, x1, y2, x2)].
	'''
	# 1. Tile boxes2 and repeate boxes1. This allows us to compare
	# every boxes1 against every boxes2 without loops.
	# TF doesn't have an equivalent to np.repeate() so simulate it
	# using tf.tile() and tf.reshape.
	b1 = np.reshape(np.tile(np.expand_dims(boxes1, 1),
							[1, 1, np.shape(boxes2)[0]]), [-1, 4])
	b2 = np.tile(boxes2, [np.shape(boxes1)[0], 1])
	# 2. Compute intersections
	b1_y1, b1_x1, b1_y2, b1_x2 = np.split(b1, 4, axis=1)
	b2_y1, b2_x1, b2_y2, b2_x2 = np.split(b2, 4, axis=1)
	y1 = np.maximum(b1_y1, b2_y1)
	x1 = np.maximum(b1_x1, b2_x1)
	y2 = np.minimum(b1_y2, b2_y2)
	x2 = np.minimum(b1_x2, b2_x2)
	intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
	# 3. Compute unions
	b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
	b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
	union = b1_area + b2_area - intersection
	# 4. Compute IoU and reshape to [boxes1, boxes2]
	iou = intersection / union
	overlaps = np.reshape(iou, [np.shape(boxes1)[0], np.shape(boxes2)[0]])
	return overlaps


def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    boxes = boxes.astype(int)
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    overlaps = np.zeros((N, K), dtype=np.float)

    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1)

                if ih > 0:
                    ua = float((boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua

    return overlaps


def hard_negative_mining(priors, ious, neg_ind):
	'''
		It used to suppress the presence of a large number of negative prediction.
	'''
	raise NotImplementedError 