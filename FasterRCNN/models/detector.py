#
# Faster R-CNN di PyTorch (tahap detektor akhir)
# Modul ini memproses proposal (RoI) dari RPN, lalu mengklasifikasikan dan meregresi bounding box.
# Box delta dihitung relatif terhadap proposal (seperti delta anchor pada RPN).
#

import torch as t
from torch import nn
from torch.nn import functional as F
from torchvision.ops import RoIPool

class DetectorNetwork(nn.Module):
  def __init__(self, num_classes, backbone):
    super().__init__()

    self._input_features = 7 * 7 * backbone.feature_map_channels

    # Definisi layer detektor
    self._roi_pool = RoIPool(output_size = (7, 7), spatial_scale = 1.0 / backbone.feature_pixels)
    self._pool_to_feature_vector = backbone.pool_to_feature_vector
    self._classifier = nn.Linear(in_features = backbone.feature_vector_size, out_features = num_classes)
    self._regressor = nn.Linear(in_features = backbone.feature_vector_size, out_features = (num_classes - 1) * 4)

    # Inisialisasi bobot layer
    self._classifier.weight.data.normal_(mean = 0.0, std = 0.01)
    self._classifier.bias.data.zero_()
    self._regressor.weight.data.normal_(mean = 0.0, std = 0.001)
    self._regressor.bias.data.zero_()

  def forward(self, feature_map, proposals):
    # Saat ini hanya mendukung batch size = 1
    assert feature_map.shape[0] == 1, "Batch size harus 1"
    batch_idxs = t.zeros((proposals.shape[0], 1)).cuda()

    # Gabungkan proposal dengan batch index: (batch_idx, x1, y1, x2, y2)
    indexed_proposals = t.cat([batch_idxs, proposals], dim=1)
    indexed_proposals = indexed_proposals[:, [0, 2, 1, 4, 3]]

    # RoI Pooling menghasilkan: (N, feature_map_channels, 7, 7)
    rois = self._roi_pool(feature_map, indexed_proposals)

    # Proses selanjutnya: feature vector -> klasifikasi & regresi box
    y = self._pool_to_feature_vector(rois=rois)
    classes_raw = self._classifier(y)
    classes = F.softmax(classes_raw, dim=1)
    box_deltas = self._regressor(y)

    return classes, box_deltas

def class_loss(predicted_classes, y_true):
  # Loss klasifikasi detektor (cross-entropy)
  epsilon = 1e-7
  scale_factor = 1.0
  cross_entropy_per_row = -(y_true * t.log(predicted_classes + epsilon)).sum(dim=1)
  N = cross_entropy_per_row.shape[0] + epsilon
  cross_entropy = t.sum(cross_entropy_per_row) / N
  return scale_factor * cross_entropy

def regression_loss(predicted_box_deltas, y_true):
  # Loss regresi detektor (smooth L1 loss)
  epsilon = 1e-7
  scale_factor = 1.0
  sigma = 1.0
  sigma_squared = sigma * sigma

  # Pisahkan mask dan target dari ground truth
  y_mask = y_true[:, 0, :]
  y_true_targets = y_true[:, 1, :]

  # Hitung loss element-wise
  x = y_true_targets - predicted_box_deltas
  x_abs = t.abs(x)
  is_negative_branch = (x_abs < (1.0 / sigma_squared)).float()
  R_negative_branch = 0.5 * x * x * sigma_squared
  R_positive_branch = x_abs - 0.5 / sigma_squared
  losses = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch

  N = y_true.shape[0] + epsilon
  relevant_loss_terms = y_mask * losses
  return scale_factor * t.sum(relevant_loss_terms) / N
