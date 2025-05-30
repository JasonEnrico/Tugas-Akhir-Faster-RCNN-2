#
# Faster R-CNN di PyTorch (fungsi-fungsi matematika bantu)
# Modul ini berisi perhitungan IoU, konversi box delta ke koordinat box absolut (baik versi NumPy maupun PyTorch).
#

import numpy as np
import torch as t

def intersection_over_union(boxes1, boxes2):
  # Menghitung IoU antar banyak pasangan box (versi NumPy)
  top_left_point = np.maximum(boxes1[:, None, 0:2], boxes2[:, 0:2])
  bottom_right_point = np.minimum(boxes1[:, None, 2:4], boxes2[:, 2:4])
  well_ordered_mask = np.all(top_left_point < bottom_right_point, axis=2)
  intersection_areas = well_ordered_mask * np.prod(bottom_right_point - top_left_point, axis=2)
  areas1 = np.prod(boxes1[:, 2:4] - boxes1[:, 0:2], axis=1)
  areas2 = np.prod(boxes2[:, 2:4] - boxes2[:, 0:2], axis=1)
  union_areas = areas1[:, None] + areas2 - intersection_areas
  epsilon = 1e-7
  return intersection_areas / (union_areas + epsilon)

def t_intersection_over_union(boxes1, boxes2):
  # Menghitung IoU antar box (versi PyTorch)
  top_left_point = t.maximum(boxes1[:, None, 0:2], boxes2[:, 0:2])
  bottom_right_point = t.minimum(boxes1[:, None, 2:4], boxes2[:, 2:4])
  well_ordered_mask = t.all(top_left_point < bottom_right_point, axis=2)
  intersection_areas = well_ordered_mask * t.prod(bottom_right_point - top_left_point, dim=2)
  areas1 = t.prod(boxes1[:, 2:4] - boxes1[:, 0:2], dim=1)
  areas2 = t.prod(boxes2[:, 2:4] - boxes2[:, 0:2], dim=1)
  union_areas = areas1[:, None] + areas2 - intersection_areas
  epsilon = 1e-7
  return intersection_areas / (union_areas + epsilon)

def convert_deltas_to_boxes(box_deltas, anchors, box_delta_means, box_delta_stds):
  # Mengubah delta box (ty, tx, th, tw) ke koordinat box absolut (versi NumPy)
  box_deltas = box_deltas * box_delta_stds + box_delta_means
  center = anchors[:, 2:4] * box_deltas[:, 0:2] + anchors[:, 0:2]
  size = anchors[:, 2:4] * np.exp(box_deltas[:, 2:4])
  boxes = np.empty(box_deltas.shape)
  boxes[:, 0:2] = center - 0.5 * size
  boxes[:, 2:4] = center + 0.5 * size
  return boxes

def t_convert_deltas_to_boxes(box_deltas, anchors, box_delta_means, box_delta_stds):
  # Mengubah delta box ke koordinat box absolut (versi PyTorch)
  box_deltas = box_deltas * box_delta_stds + box_delta_means
  center = anchors[:, 2:4] * box_deltas[:, 0:2] + anchors[:, 0:2]
  size = anchors[:, 2:4] * t.exp(box_deltas[:, 2:4])
  boxes = t.empty(box_deltas.shape, dtype=t.float32, device="cuda")
  boxes[:, 0:2] = center - 0.5 * size
  boxes[:, 2:4] = center + 0.5 * size
  return boxes
