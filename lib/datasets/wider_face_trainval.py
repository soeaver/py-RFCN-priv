# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg

class wider_face_trainval(imdb):
    def __init__(self):
        imdb.__init__(self, 'wider_face_trainval')
        self.data_root = os.path.join(cfg.DATABASE_ROOT, 'Wider_Face/')
        self.source = os.path.join(cfg.DATABASE_ROOT, 'Wider_Face/wider_face_annotation_refine', 'wider_trainval_img2xml.txt')
        self.img_set = os.path.join(cfg.DATABASE_ROOT, 'Wider_Face/wider_face_annotation_refine', 'wider_trainval.txt')

        assert os.path.exists(self.data_root), \
                'Data root path does not exist: {}'.format(self.data_root)
        assert os.path.exists(self.source), \
                'Source file does not exist: {}'.format(self.source)
        assert os.path.exists(self.img_set), \
                'Image set file does not exist: {}'.format(self.img_set)

        self._classes = ('__background__', # always index 0
                         'face')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_index = self._load_image_set_index
        self.image_path_list = []
        self.xml_path_list = []
        self._load_image_xml_path()
        self._roidb_handler = self.gt_roidb

        # PASCAL specific config options
        self.config = {'cleanup'     : False,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

    def _load_image_xml_path(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
	f = open(self.source, 'r')
        for i in f:
            self.image_path_list.append(self.data_root + i.strip().split(' ')[0])
            self.xml_path_list.append(self.data_root + i.strip().split(' ')[1])
        f.close()

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        p = self._image_index.index(index)
        image_path = os.path.join(self.image_path_list[p])

        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    @property
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        with open(self.img_set) as f:
            image_index = [x.strip() for x in f.readlines()]

        return image_index

    def _load_pascal_annotation(self, xml_path):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        tree = ET.parse(xml_path)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = max(float(bbox.find('xmin').text), 1) - 1
            y1 = max(float(bbox.find('ymin').text), 1) - 1
            x2 = max(float(bbox.find('xmax').text), 1) - 1
            y2 = max(float(bbox.find('ymax').text), 1) - 1
            if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0:
                print [x1, y1, x2, y2]
                continue
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
  	#print self.image_index 
        gt_roidb = [self._load_pascal_annotation(xml_path)
                    for xml_path in self.xml_path_list]
   
        return gt_roidb


