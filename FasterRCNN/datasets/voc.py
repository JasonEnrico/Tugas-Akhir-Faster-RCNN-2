from dataclasses import dataclass
import numpy as np
import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET
from typing import List
from typing import Tuple

from .training_sample import Box
from .training_sample import TrainingSample
from . import image
from pytorch.FasterRCNN.models import anchors


class Dataset:
  """
  A VOC dataset iterator for a particular split (train, val, etc.)
  """

  num_classes = 3
  class_index_to_name = {
    0:  "background",
    1:  "no_helmet",
    2:  "helmet"
  }

  def __init__(self, split, image_preprocessing_params, compute_feature_map_shape_fn, feature_pixels = 16, dir = "VOCdevkit/VOC2007", augment = True, shuffle = True, allow_difficult = False, cache = True):
    """
    Parameters
    ----------
    split : str
      Dataset split to load: train, val, or trainval.
    image_preprocessing_params : dataset.image.PreprocessingParams
      Image preprocessing parameters to apply when loading images.
    compute_feature_map_shape_fn : Callable[Tuple[int, int, int], Tuple[int, int, int]]
      Function to compute feature map shape, (channels, height, width), from
      input image shape, (channels, height, width).
    feature_pixels : int
      Size of each cell in the Faster R-CNN feature map in image pixels. This
      is the separation distance between anchors.
    dir : str
      Root directory of dataset.
    augment : bool
      Whether to randomly augment (horizontally flip) images during iteration
      with 50% probability.
    shuffle : bool
      Whether to shuffle the dataset each time it is iterated.
    allow_difficult : bool
      Whether to include ground truth boxes that are marked as "difficult".
    cache : bool
      Whether to training samples in memory after first being generated.
    """
    if not os.path.exists(dir):
      raise FileNotFoundError("Dataset directory does not exist: %s" % dir)
    self.split = split
    self._dir = dir
    self.class_index_to_name = self._get_classes()
    self.class_name_to_index = { class_name: class_index for (class_index, class_name) in self.class_index_to_name.items() }
    self.num_classes = len(self.class_index_to_name)
    # assert self.num_classes == Dataset.num_classes, "Dataset does not have the expected number of classes (found %d but expected %d)" % (self.num_classes, Dataset.num_classes)
    # assert self.class_index_to_name == Dataset.class_index_to_name, "Dataset does not have the expected class mapping"
    self._filepaths = self._get_filepaths()
    self.num_samples = len(self._filepaths)
    self._gt_boxes_by_filepath = self._get_ground_truth_boxes(filepaths = self._filepaths, allow_difficult = allow_difficult)
    self._i = 0
    self._iterable_filepaths = self._filepaths.copy()
    self._image_preprocessing_params = image_preprocessing_params
    self._compute_feature_map_shape_fn = compute_feature_map_shape_fn
    self._feature_pixels = feature_pixels
    self._augment = augment
    self._shuffle = shuffle
    self._cache = cache
    self._unaugmented_cached_sample_by_filepath = {}
    self._augmented_cached_sample_by_filepath = {}

  def __iter__(self):
    self._i = 0
    if self._shuffle:
      random.shuffle(self._iterable_filepaths)
    return self

  def __next__(self):
    if self._i >= len(self._iterable_filepaths):
      raise StopIteration

    # Next file to load
    filepath = self._iterable_filepaths[self._i]
    self._i += 1

    # Augment?
    flip = random.randint(0, 1) != 0 if self._augment else 0
    cached_sample_by_filepath = self._augmented_cached_sample_by_filepath if flip else self._unaugmented_cached_sample_by_filepath

    # Load and, if caching, write back to cache
    if filepath in cached_sample_by_filepath:
      sample = cached_sample_by_filepath[filepath]
    else:
      sample = self._generate_training_sample(filepath = filepath, flip = flip)
    if self._cache:
      cached_sample_by_filepath[filepath] = sample

    # Return the sample
    return sample

  def _generate_training_sample(self, filepath, flip):
    # Load and preprocess the image
    scaled_image_data, scaled_image, scale_factor, original_shape = image.load_image(url = filepath, preprocessing = self._image_preprocessing_params, min_dimension_pixels = 600, horizontal_flip = flip)
    _, original_height, original_width = original_shape

    # Scale ground truth boxes to new image size
    scaled_gt_boxes = []
    for box in self._gt_boxes_by_filepath[filepath]:
      if flip:
        corners = np.array([
          box.corners[0],
          original_width - 1 - box.corners[3],
          box.corners[2],
          original_width - 1 - box.corners[1]
        ])
      else:
        corners = box.corners
      scaled_box = Box(
        class_index = box.class_index,
        class_name = box.class_name,
        corners = corners * scale_factor
      )
      scaled_gt_boxes.append(scaled_box)

    # Generate anchor maps and RPN truth map
    anchor_map, anchor_valid_map = anchors.generate_anchor_maps(image_shape = scaled_image_data.shape, feature_map_shape = self._compute_feature_map_shape_fn(scaled_image_data.shape), feature_pixels = self._feature_pixels)
    gt_rpn_map, gt_rpn_object_indices, gt_rpn_background_indices = anchors.generate_rpn_map(anchor_map = anchor_map, anchor_valid_map = anchor_valid_map, gt_boxes = scaled_gt_boxes)

    # Return sample
    return TrainingSample(
      anchor_map = anchor_map,
      anchor_valid_map = anchor_valid_map,
      gt_rpn_map = gt_rpn_map,
      gt_rpn_object_indices = gt_rpn_object_indices,
      gt_rpn_background_indices = gt_rpn_background_indices,
      gt_boxes = scaled_gt_boxes,
      image_data = scaled_image_data,
      image = scaled_image,
      filepath = filepath
    )

  def _get_classes(self):
    # Ambil semua nama file dari split (train.txt, val.txt, dll)
    image_list_file = os.path.join(self._dir, "ImageSets", "Main", self.split + ".txt")
    with open(image_list_file) as f:
        basenames = [line.strip() for line in f.readlines()]

    classes = set()
    for basename in basenames:
        annotation_path = os.path.join(self._dir, "Annotations", basename + ".xml")
        if not os.path.exists(annotation_path):
            continue
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip().lower()
            classes.add(class_name)

    print(f"Classes ditemukan dari split '{self.split}':", classes)
    if len(classes) == 0:
        print(f"[WARNING] Tidak ada class ditemukan di split '{self.split}', lanjutkan tetap...")

    class_index_to_name = { (1 + i): name for i, name in enumerate(sorted(classes)) }
    class_index_to_name[0] = "background"
    return class_index_to_name

  def _get_filepaths(self):
    image_list_file = os.path.join(self._dir, "ImageSets", "Main", self.split + ".txt")
    with open(image_list_file) as fp:
      basenames = [ line.strip() for line in fp.readlines() ] # strip newlines
    image_paths = [ os.path.join(self._dir, "JPEGImages", basename) + ".jpg" for basename in basenames ]
    return image_paths

  def _get_ground_truth_boxes(self, filepaths, allow_difficult):
    gt_boxes_by_filepath = {}
    for filepath in filepaths:
      basename = os.path.splitext(os.path.basename(filepath))[0]
      annotation_file = os.path.join(self._dir, "Annotations", basename) + ".xml"
      tree = ET.parse(annotation_file)
      root = tree.getroot()
      assert tree != None, "Failed to parse %s" % annotation_file
      assert len(root.findall("size")) == 1
      size = root.find("size")
      assert len(size.findall("depth")) == 1
      depth = int(size.find("depth").text)
      assert depth == 3
      boxes = []
      for obj in root.findall("object"):
        assert len(obj.findall("name")) == 1
        assert len(obj.findall("bndbox")) == 1
        difficult_node = obj.find("difficult")
        is_difficult = False
        if difficult_node is not None:
            is_difficult = int(difficult_node.text) != 0
        if is_difficult and not allow_difficult:
            continue

        if is_difficult and not allow_difficult:
          continue  # ignore difficult examples unless asked to include them
        class_name = obj.find("name").text.strip().lower()
        bndbox = obj.find("bndbox")
        assert len(bndbox.findall("xmin")) == 1
        assert len(bndbox.findall("ymin")) == 1
        assert len(bndbox.findall("xmax")) == 1
        assert len(bndbox.findall("ymax")) == 1
        x_min = int(bndbox.find("xmin").text) - 1  # convert to 0-based pixel coordinates
        y_min = int(bndbox.find("ymin").text) - 1
        x_max = int(bndbox.find("xmax").text) - 1
        y_max = int(bndbox.find("ymax").text) - 1
        corners = np.array([ y_min, x_min, y_max, x_max ]).astype(np.float32)
        box = Box(class_index = self.class_name_to_index[class_name], class_name = class_name, corners = corners)
        boxes.append(box)
      assert len(boxes) > 0
      gt_boxes_by_filepath[filepath] = boxes
    return gt_boxes_by_filepath
