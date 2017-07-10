# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms

DEBUG = False

class IterProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._feat_stride = layer_params['feat_stride']
        add_old_rois = layer_params.get('add_old_rois', False)
        iter_thresh = layer_params.get('iter_thresh', 0.05)

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        rois = bottom[0].data	# rois
        scores = bottom[1].data	# cls_prob
        bbox_deltas = bottom[2].data	# bbox_pred
        im_info = bottom[3].data[0, :]
        boxes = rois[:, 1:5] / im_info[2]

        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, (im_info[0], im_info[1]))

        scores_old = scores.copy()
        pred_boxes_old = pred_boxes.copy()

        keep = []
        for j in xrange(len(scores)):
            for k in xrange(0, num_classes):
                if scores[j][k] > iter_thresh:
                    keep.append(pred_boxes[j][k * 4:(k + 1) * 4])

        keep = np.asarray(keep)
        if len(keep) == 0:
            continue

        im_scales_temp = np.tile(0, (len(keep), 1))
        keep = keep * im_info[2]
        iter_roi_blob = np.hstack((im_scales_temp, keep)).astype(np.float32, copy=False)

        top[0].reshape(*(iter_roi_blob.shape))
        top[0].data[...] = iter_roi_blob


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


