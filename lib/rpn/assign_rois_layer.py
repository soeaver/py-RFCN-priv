# --------------------------------------------------------
# FPN
# Copyright (c) 2017 BUPT-PRIV
# Licensed under The MIT License [see LICENSE for details]
# Written by Soeaver Yang
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg

def assign_pyramid(roi, k0=4, size=224):
    roi_width = roi[3] - roi[1]
    roi_height = roi[4] - roi[2]

    return np.ceil(np.log2(np.sqrt(float(roi_width*roi_height))/float(size)) + k0)


class AssignROISLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """
    
    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._pyramid_number = layer_params.get('pyramid_number', ((2, 3, 4, 5)))
        self._base_size = layer_params.get('base_scale', 4)
        self._pre_training_size = layer_params.get('pre_training_size', 224) # 299 for inception

        assert len(top) == len(self._pyramid_number)

        for i in xrange(len(top)):
            top[i].reshape(1, 5)

    def forward(self, bottom, top):
        all_rois = bottom[0].data
        min_pyramid = min(self._pyramid_number)
        max_pyramid = max(self._pyramid_number)

        assigned_rois = [[] for _ in xrange(len(self._pyramid_number))]  # 2, 3, 4, 5
        for _ in all_rois:
            k = assign_pyramid(_, k0=self._base_size, size=self._pre_training_size)
            k = min(max(min_pyramid, k), max_pyramid)
            idx = self._pyramid_number.index(k)
            assigned_rois[idx].append(_)

        for i in xrange(len(self._pyramid_number)):
            rois_blob = np.asarray(assigned_rois[i])
            top[i].reshape(*(rois_blob.shape))
            top[i].data[...] = rois_blob

        # print top[0].data[...].shape

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
