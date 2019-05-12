import collections
from collections import namedtuple
import numpy as np
import itertools


BoxSizes = collections.namedtuple('BoxSizes', ['min', 'max'])
Spec = collections.namedtuple(
	'Spec',
	['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])
specs = [
	Spec(40, 10, BoxSizes(130, 160), [2]),
	Spec(20, 21, BoxSizes(160, 211), [2, 3]),
	Spec(10, 42, BoxSizes(211, 262), [2, 3]),
	Spec(5, 83, BoxSizes(262, 313), [2, 3]),
	Spec(3, 139, BoxSizes(313, 364), [2]),
	Spec(1, 416, BoxSizes(364, 415), [2])
]

def anchors(specs=specs, image_size=416, clip=True):
	"""Generate Prior Boxes.
	Args:
		specs: Specs about the shapes of sizes of prior boxes. i.e.
		image_size: image size.

	Returns:
		priors: a list of priors: [[center_x, center_y, h, w]]. All the values
			are relative to the image size (416x416).
	"""
	boxes = []
	for spec in specs:
		scale = image_size / spec.shrinkage
		for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
			x_center = (i + 0.5) / scale
			y_center = (j + 0.5) / scale

			# small sized square box
			size = spec.box_sizes.min
			h = w = size / image_size
			boxes.append([
				x_center,
				y_center,
				w,
				h
			])

			# big sized square box
			size = np.sqrt(spec.box_sizes.max * spec.box_sizes.min)
			h = w = size / image_size
			boxes.append([
				x_center,
				y_center,
				w,
				h
			])

			# change h/w ratio of the small sized box
			# based on the SSD implementation,
			# it only applies ratio to the smallest size.
			# it looks wierd.
			size = spec.box_sizes.min
			h = w = size / image_size
			for ratio in spec.aspect_ratios:
				ratio = np.sqrt(ratio)
				boxes.append([
					x_center,
					y_center,
					w / ratio,
					h * ratio
				])
				boxes.append([
					x_center,
					y_center,
					w * ratio,
					h / ratio
				])
	boxes = np.array(boxes)
	if clip:
		boxes = np.clip(boxes, 0.0, 1.0)
	return boxes

if __name__ == "__main__":
	boxes = generate_ssd_priors(specs)
	print (boxes)
	print (boxes.shape)