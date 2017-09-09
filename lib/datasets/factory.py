# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import numpy as np
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco

from datasets.voc_0712_trainval import voc_0712_trainval
from datasets.voc_07trainvaltest_12trainval import voc_07trainvaltest_12trainval
from datasets.coco21trainval_voc0712trainval import coco21trainval_voc0712trainval

from datasets.coco81_trainval35k import coco81_trainval35k
from datasets.coco81_trainval import coco81_trainval
from datasets.mscoco2017_81trainval import mscoco2017_81trainval

from datasets.wider_face_trainval import wider_face_trainval

from datasets.ilsvrc2017_det_train import ilsvrc2017_det_train



__sets['voc_0712_trainval'] = voc_0712_trainval
__sets['voc_07trainvaltest_12trainval'] = voc_07trainvaltest_12trainval
__sets['coco21trainval_voc0712trainval'] = coco21trainval_voc0712trainval
__sets['coco81_trainval35k'] = coco81_trainval35k
__sets['coco81_trainval'] = coco81_trainval
__sets['mscoco2017_81trainval'] = mscoco2017_81trainval
__sets['wider_face_trainval'] = wider_face_trainval
__sets['ilsvrc2017_det_train'] = ilsvrc2017_det_train


# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

print list_imdbs()
