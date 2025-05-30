import numpy as np
import torch as t
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms

from . import math_utils

class RegionProposalNetwork(nn.Module):
  def __init__(self, feature_map_channels):
    super().__init__()

    # Jumlah anchor (skala dan rasio kombinasi)
    num_anchors = 9
    channels = feature_map_channels

    # Layer RPN
    self._rpn_conv1 = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, padding = "same")
    self._rpn_class = nn.Conv2d(in_channels = channels, out_channels = num_anchors, kernel_size = 1, stride = 1, padding = "same")
    self._rpn_boxes = nn.Conv2d(in_channels = channels, out_channels = num_anchors * 4, kernel_size = 1, stride = 1, padding = "same")

    # Inisialisasi bobot layer
    self._rpn_conv1.weight.data.normal_(mean = 0.0, std = 0.01)
    self._rpn_conv1.bias.data.zero_()
    self._rpn_class.weight.data.normal_(mean = 0.0, std = 0.01)
    self._rpn_class.bias.data.zero_()
    self._rpn_boxes.weight.data.normal_(mean = 0.0, std = 0.01)
    self._rpn_boxes.bias.data.zero_()

  def forward(self, feature_map, image_shape, anchor_map, anchor_valid_map, max_proposals_pre_nms, max_proposals_post_nms):
    # Proses feature map melalui layer RPN
    y = F.relu(self._rpn_conv1(feature_map))
    objectness_score_map = t.sigmoid(self._rpn_class(y))
    box_deltas_map = self._rpn_boxes(y)

    # Ubah dimensi tensor agar memudahkan pengolahan
    objectness_score_map = objectness_score_map.permute(0, 2, 3, 1).contiguous()
    box_deltas_map = box_deltas_map.permute(0, 2, 3, 1).contiguous()

    # Ekstraksi anchor dan prediksi valid
    anchors, objectness_scores, box_deltas = self._extract_valid(
      anchor_map, anchor_valid_map, objectness_score_map, box_deltas_map
    )

    box_deltas = box_deltas.detach()

    # Konversi delta box ke koordinat box aktual
    proposals = math_utils.t_convert_deltas_to_boxes(
      box_deltas = box_deltas,
      anchors = t.from_numpy(anchors).cuda(),
      box_delta_means = t.tensor([0, 0, 0, 0], dtype = t.float32, device = "cuda"),
      box_delta_stds = t.tensor([1, 1, 1, 1], dtype = t.float32, device = "cuda")
    )

    # Seleksi top-N proposal terbaik sebelum NMS
    sorted_indices = t.argsort(objectness_scores).flip(dims = (0,))
    proposals = proposals[sorted_indices][:max_proposals_pre_nms]
    objectness_scores = objectness_scores[sorted_indices][:max_proposals_pre_nms]

    # Kliping ke batas gambar
    proposals[:,0:2] = t.clamp(proposals[:,0:2], min = 0)
    proposals[:,2] = t.clamp(proposals[:,2], max = image_shape[1])
    proposals[:,3] = t.clamp(proposals[:,3], max = image_shape[2])

    # Buang proposal dengan sisi < 16 piksel
    height = proposals[:,2] - proposals[:,0]
    width = proposals[:,3] - proposals[:,1]
    idxs = t.where((height >= 16) & (width >= 16))[0]
    proposals = proposals[idxs]
    objectness_scores = objectness_scores[idxs]

    # Non-Maximum Suppression (NMS)
    idxs = nms(proposals, objectness_scores, iou_threshold = 0.7)
    idxs = idxs[:max_proposals_post_nms]
    proposals = proposals[idxs]

    return objectness_score_map, box_deltas_map, proposals

  def _extract_valid(self, anchor_map, anchor_valid_map, objectness_score_map, box_deltas_map):
    assert objectness_score_map.shape[0] == 1

    height, width, num_anchors = anchor_valid_map.shape
    anchors = anchor_map.reshape((height * width * num_anchors, 4))
    anchors_valid = anchor_valid_map.reshape((height * width * num_anchors))
    scores = objectness_score_map.reshape((height * width * num_anchors))
    box_deltas = box_deltas_map.reshape((height * width * num_anchors, 4))

    idxs = anchors_valid > 0
    return anchors[idxs], scores[idxs], box_deltas[idxs]

def class_loss(predicted_scores, y_true):
  # Loss klasifikasi RPN (binary cross entropy)
  epsilon = 1e-7
  y_true_class = y_true[:,:,:,:,1].reshape(predicted_scores.shape)
  y_mask = y_true[:,:,:,:,0].reshape(predicted_scores.shape)

  N_cls = t.count_nonzero(y_mask) + epsilon
  loss_all_anchors = F.binary_cross_entropy(predicted_scores, y_true_class, reduction = "none")
  relevant_loss_terms = y_mask * loss_all_anchors

  return t.sum(relevant_loss_terms) / N_cls

def regression_loss(predicted_box_deltas, y_true):
  # Loss regresi RPN (smooth L1 loss)
  epsilon = 1e-7
  scale_factor = 1.0
  sigma = 3.0
  sigma_squared = sigma * sigma

  y_true_regression = y_true[:,:,:,:,2:6].reshape(predicted_box_deltas.shape)
  y_included = y_true[:,:,:,:,0].reshape(y_true.shape[0:4])
  y_positive = y_true[:,:,:,:,1].reshape(y_true.shape[0:4])
  y_mask = y_included * y_positive
  y_mask = y_mask.repeat_interleave(4, dim = 3)

  N_cls = t.count_nonzero(y_included) + epsilon

  x = y_true_regression - predicted_box_deltas
  x_abs = t.abs(x)
  is_negative_branch = (x_abs < (1.0 / sigma_squared)).float()
  R_negative_branch = 0.5 * x * x * sigma_squared
  R_positive_branch = x_abs - 0.5 / sigma_squared
  loss_all_anchors = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch

  relevant_loss_terms = y_mask * loss_all_anchors
  return scale_factor * t.sum(relevant_loss_terms) / N_cls
