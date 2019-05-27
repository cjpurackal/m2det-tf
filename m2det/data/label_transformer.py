import numpy as np
from m2det.utils import generate_anchors, iou

def transformer1(boxes, num_classes, iou_thresh):
	anchors = generate_anchors()
	num_classes += 1
	
	def box_generator():
		for box in boxes[:,:4]:
			encoded_box = np.zeros((len(anchors), 4 + 1))
			iou_scores = iou(anchors, box)
			iou_mask = iou_scores > iou_thresh
			picked_iou = iou_scores[iou_mask]
			picked_anchors = anchors[iou_mask]
			encoded_box[:, -1][iou_mask] = picked_iou
			#computing prior-box center offsets, log
			box_wh = box[2:] - box[:2]
			picked_anchors_wh = picked_anchors[:,2:4] - picked_anchors[:, :2]
			box_centre_xy = 0.5 *(box[:2] + box[2:])
			picked_anchors_centre_xy = 0.5 * (picked_anchors[:,:2] + picked_anchors[:, 2:4])
			
			encoded_box[:,:2][iou_mask] = box_centre_xy - picked_anchors_centre_xy
			encoded_box[:,:2][iou_mask] /= picked_anchors_wh
			encoded_box[:, :2][iou_mask] /= .1 
			encoded_box[:, 2:4][iou_mask] = np.log(box_wh/ picked_anchors_wh) 
			encoded_box[:, 2:4][iou_mask] /= .2
			yield encoded_box.ravel()

	encoded_boxes = [encoded_box for encoded_box in box_generator()]
	encoded_boxes = np.reshape(encoded_boxes,(-1, len(anchors), 5))
	truth_tensor = np.zeros((len(anchors), 4 + num_classes + 1))
	truth_tensor[:, 4] = 1.0 # background
	best_iou = encoded_boxes[:, :, -1].max(axis=0)
	best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
	best_iou_mask = best_iou > 0 # judge by iou between prior and bbox
	best_iou_idx = best_iou_idx[best_iou_mask]
	assign_num = len(best_iou_idx)
	encoded_boxes = encoded_boxes[:, best_iou_mask, :]
	truth_tensor[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
	truth_tensor[:, 4][best_iou_mask] = 0 # background
	truth_tensor[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]
	truth_tensor[:, -1][best_iou_mask] = 1 # objectness
	return truth_tensor