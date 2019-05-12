import numpy as np
from utils import generate
from utils import misc


class Anchors:
    def __init__(self, box, config):
        self.image_size = config["model"]["input_size"]
        # :\ temp ground truth
        self.gt_box = box / self.image_size

    def call(self):
        priors = generate.anchors()
        priors = misc.bbox_to_xml(priors)
        # :\ temp ground truth
        ious = misc.bbox_overlaps(priors, np.expand_dims(np.array(self.gt_box), axis=0))
        pos_ind = [
                    i for i, x in enumerate(ious)
                    if x > 0.5
                    ]
        neg_ind = [
                    i for i, x in enumerate(ious)
                    if x < 0.5
                    ]
        # TO DO
        # neg_ind = misc.hard_negative_mining(priors, ious, neg_ind)
        return priors, pos_ind, neg_ind
