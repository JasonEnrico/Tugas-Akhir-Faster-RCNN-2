#
# Faster R-CNN di PyTorch (anchor generator)
# Modul ini menghasilkan anchor box, label ground truth untuk training RPN, dan map valid anchor.
#

import itertools
from math import sqrt
import numpy as np
from . import math_utils

def _compute_anchor_sizes():
  # Menghitung kombinasi skala dan rasio anchor
  areas = [128*128, 256*256, 512*512]
  x_aspects = [0.5, 1.0, 2.0]
  heights = np.array([x_aspects[j] * sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3))])
  widths = np.array([sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3))])
  return np.vstack([heights, widths]).T

def generate_anchor_maps(image_shape, feature_map_shape, feature_pixels):
  # Membuat anchor map berdasarkan ukuran gambar dan feature map
  assert len(image_shape) == 3
  anchor_sizes = _compute_anchor_sizes()
  num_anchors = anchor_sizes.shape[0]

  # Template anchor (titik tengah feature map di-offset oleh ukuran anchor)
  anchor_template = np.empty((num_anchors, 4))
  anchor_template[:, 0:2] = -0.5 * anchor_sizes
  anchor_template[:, 2:4] = +0.5 * anchor_sizes

  height = feature_map_shape[-2]
  width = feature_map_shape[-1]
  y_cell_coords = np.arange(height)
  x_cell_coords = np.arange(width)
  cell_coords = np.array(np.meshgrid(y_cell_coords, x_cell_coords)).transpose([2, 1, 0])
  center_points = cell_coords * feature_pixels + 0.5 * feature_pixels

  center_points = np.tile(center_points, reps=2)
  center_points = np.tile(center_points, reps=num_anchors)
  anchors = center_points.astype(np.float32) + anchor_template.flatten()
  anchors = anchors.reshape((height * width * num_anchors, 4))

  image_height, image_width = image_shape[1:]
  valid = np.all((anchors[:, 0:2] >= [0, 0]) & (anchors[:, 2:4] <= [image_height, image_width]), axis=1)

  anchor_map = np.empty((anchors.shape[0], 4))
  anchor_map[:, 0:2] = 0.5 * (anchors[:, 0:2] + anchors[:, 2:4])
  anchor_map[:, 2:4] = anchors[:, 2:4] - anchors[:, 0:2]
  anchor_map = anchor_map.reshape((height, width, num_anchors * 4))
  anchor_valid_map = valid.reshape((height, width, num_anchors))
  return anchor_map.astype(np.float32), anchor_valid_map.astype(np.float32)

def generate_rpn_map(anchor_map, anchor_valid_map, gt_boxes, object_iou_threshold=0.7, background_iou_threshold=0.3):
  # Membuat ground truth map untuk training RPN
  height, width, num_anchors = anchor_valid_map.shape
  gt_box_corners = np.array([box.corners for box in gt_boxes])
  num_gt_boxes = len(gt_boxes)

  gt_box_centers = 0.5 * (gt_box_corners[:, 0:2] + gt_box_corners[:, 2:4])
  gt_box_sides = gt_box_corners[:, 2:4] - gt_box_corners[:, 0:2]

  anchor_map = anchor_map.reshape((-1, 4))
  anchors = np.empty(anchor_map.shape)
  anchors[:, 0:2] = anchor_map[:, 0:2] - 0.5 * anchor_map[:, 2:4]
  anchors[:, 2:4] = anchor_map[:, 0:2] + 0.5 * anchor_map[:, 2:4]
  n = anchors.shape[0]

  objectness_score = np.full(n, -1)
  gt_box_assignments = np.full(n, -1)
  ious = math_utils.intersection_over_union(boxes1=anchors, boxes2=gt_box_corners)
  ious[anchor_valid_map.flatten() == 0, :] = -1.0

  max_iou_per_anchor = np.max(ious, axis=1)
  best_box_idx_per_anchor = np.argmax(ious, axis=1)
  max_iou_per_gt_box = np.max(ious, axis=0)
  highest_iou_anchor_idxs = np.where(ious == max_iou_per_gt_box)[0]

  objectness_score[max_iou_per_anchor < background_iou_threshold] = 0
  objectness_score[max_iou_per_anchor >= object_iou_threshold] = 1
  objectness_score[highest_iou_anchor_idxs] = 1
  gt_box_assignments[:] = best_box_idx_per_anchor
  enable_mask = (objectness_score >= 0).astype(np.float32)
  objectness_score[objectness_score < 0] = 0

  box_delta_targets = np.empty((n, 4))
  box_delta_targets[:, 0:2] = (gt_box_centers[gt_box_assignments] - anchor_map[:, 0:2]) / anchor_map[:, 2:4]
  box_delta_targets[:, 2:4] = np.log(gt_box_sides[gt_box_assignments] / anchor_map[:, 2:4])

  rpn_map = np.zeros((height, width, num_anchors, 6))
  rpn_map[:, :, :, 0] = anchor_valid_map * enable_mask.reshape((height, width, num_anchors))
  rpn_map[:, :, :, 1] = objectness_score.reshape((height, width, num_anchors))
  rpn_map[:, :, :, 2:6] = box_delta_targets.reshape((height, width, num_anchors, 4))

  rpn_map_coords = np.transpose(np.mgrid[0:height, 0:width, 0:num_anchors], (1, 2, 3, 0))
  object_anchor_idxs = rpn_map_coords[np.where((rpn_map[:, :, :, 1] > 0) & (rpn_map[:, :, :, 0] > 0))]
  background_anchor_idxs = rpn_map_coords[np.where((rpn_map[:, :, :, 1] == 0) & (rpn_map[:, :, :, 0] > 0))]

  return rpn_map.astype(np.float32), object_anchor_idxs, background_anchor_idxs
