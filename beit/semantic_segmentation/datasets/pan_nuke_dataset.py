from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

import os.path as osp
import numpy as np
from PIL import Image


@DATASETS.register_module()
class PanNukeDataset(CustomDataset):
  classes = ('neoplastic ', 'inflammatory', 'soft', 'dead', 'epithelial')

  CLASSES = classes
  PALETTE = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], [0, 11, 123]]

  def __init__(self, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', split=None, **kwargs)
    assert osp.exists(self.img_dir)