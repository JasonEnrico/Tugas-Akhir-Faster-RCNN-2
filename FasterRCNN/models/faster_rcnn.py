#
# Faster R-CNN di PyTorch (modul utama training dan inference)
# Modul ini mengintegrasikan semua tahapan Faster R-CNN: feature extractor, RPN, detektor, sampling anchor/proposal, training dan inferensi.
#

from dataclasses import dataclass
import numpy as np
import random
import torch as t
from torch import nn
from torchvision.ops import nms

from pytorch.FasterRCNN import utils
from . import anchors
from . import math_utils
from . import rpn
from . import detector

class FasterRCNNModel(nn.Module):
  @dataclass
  class Loss:
    rpn_class: float
    rpn_regression: float
    detector_class: float
    detector_regression: float
    total: float

  def __init__(self, num_classes, backbone, rpn_minibatch_size=256, proposal_batch_size=128, allow_edge_proposals=True):
    super().__init__()

    self._num_classes = num_classes
    self._rpn_minibatch_size = rpn_minibatch_size
    self._proposal_batch_size = proposal_batch_size
    self._detector_box_delta_means = [0, 0, 0, 0]
    self._detector_box_delta_stds = [0.1, 0.1, 0.2, 0.2]

    self.backbone = backbone

    self._stage1_feature_extractor = backbone.feature_extractor
    self._stage2_region_proposal_network = rpn.RegionProposalNetwork(
      feature_map_channels=backbone.feature_map_channels,
      allow_edge_proposals=allow_edge_proposals
    )
    self._stage3_detector_network = detector.DetectorNetwork(
      num_classes=num_classes,
      backbone=backbone
    )

  def forward(self, image_data, anchor_map=None, anchor_valid_map=None):
    assert image_data.shape[0] == 1, "Batch size harus 1"
    image_shape = image_data.shape[1:]

    if anchor_map is None or anchor_valid_map is None:
      feature_map_shape = self.backbone.compute_feature_map_shape(image_shape=image_shape)
      anchor_map, anchor_valid_map = anchors.generate_anchor_maps(
        image_shape=image_shape, feature_map_shape=feature_map_shape, feature_pixels=self.backbone.feature_pixels)

    feature_map = self._stage1_feature_extractor(image_data=image_data)
    objectness_score_map, box_deltas_map, proposals = self._stage2_region_proposal_network(
      feature_map=feature_map,
      image_shape=image_shape,
      anchor_map=anchor_map,
      anchor_valid_map=anchor_valid_map,
      max_proposals_pre_nms=6000,
      max_proposals_post_nms=300
    )
    classes, box_deltas = self._stage3_detector_network(
      feature_map=feature_map,
      proposals=proposals
    )

    return proposals, classes, box_deltas

  @utils.no_grad
  def predict(self, image_data, score_threshold, anchor_map=None, anchor_valid_map=None):
    self.eval()
    assert image_data.shape[0] == 1, "Batch size harus 1"

    proposals, classes, box_deltas = self(
      image_data=image_data,
      anchor_map=anchor_map,
      anchor_valid_map=anchor_valid_map
    )
    proposals = proposals.cpu().numpy()
    classes = classes.cpu().numpy()
    box_deltas = box_deltas.cpu().numpy()

    proposal_anchors = np.empty(proposals.shape)
    proposal_anchors[:, 0] = 0.5 * (proposals[:, 0] + proposals[:, 2])
    proposal_anchors[:, 1] = 0.5 * (proposals[:, 1] + proposals[:, 3])
    proposal_anchors[:, 2:4] = proposals[:, 2:4] - proposals[:, 0:2]

    boxes_and_scores_by_class_idx = {}
    for class_idx in range(1, classes.shape[1]):
      box_delta_idx = (class_idx - 1) * 4
      box_delta_params = box_deltas[:, (box_delta_idx):(box_delta_idx + 4)]
      proposal_boxes_this_class = math_utils.convert_deltas_to_boxes(
        box_deltas=box_delta_params,
        anchors=proposal_anchors,
        box_delta_means=self._detector_box_delta_means,
        box_delta_stds=self._detector_box_delta_stds)

      proposal_boxes_this_class[:, 0::2] = np.clip(proposal_boxes_this_class[:, 0::2], 0, image_data.shape[2] - 1)
      proposal_boxes_this_class[:, 1::2] = np.clip(proposal_boxes_this_class[:, 1::2], 0, image_data.shape[3] - 1)

      scores_this_class = classes[:, class_idx]
      sufficiently_scoring_idxs = np.where(scores_this_class > score_threshold)[0]
      proposal_boxes_this_class = proposal_boxes_this_class[sufficiently_scoring_idxs]
      scores_this_class = scores_this_class[sufficiently_scoring_idxs]
      boxes_and_scores_by_class_idx[class_idx] = (proposal_boxes_this_class, scores_this_class)

    scored_boxes_by_class_idx = {}
    for class_idx, (boxes, scores) in boxes_and_scores_by_class_idx.items():
      idxs = nms(
        boxes=t.from_numpy(boxes).cuda(),
        scores=t.from_numpy(scores).cuda(),
        iou_threshold=0.3
      ).cpu().numpy()
      boxes = boxes[idxs]
      scores = np.expand_dims(scores[idxs], axis=0)
      scored_boxes = np.hstack([boxes, scores.T])
      scored_boxes_by_class_idx[class_idx] = scored_boxes

    return scored_boxes_by_class_idx
