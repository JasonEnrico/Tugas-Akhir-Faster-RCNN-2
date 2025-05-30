#
# Modul Perhitungan Statistik Training dan Evaluasi Model Faster R-CNN
# File: statistics.py
#
# Berisi class untuk menghitung statistik selama training (loss) dan evaluasi (precision-recall, mAP)
# Digunakan pada tahap training maupun validasi model.
#

from collections import defaultdict
import numpy as np
from .models.math_utils import intersection_over_union

# Class utama untuk menghitung statistik training (loss)
class TrainingStatistics:
    def __init__(self):
        self.rpn_class_loss = float("inf")
        self.rpn_regression_loss = float("inf")
        self.detector_class_loss = float("inf")
        self.detector_regression_loss = float("inf")
        self._rpn_class_losses = []
        self._rpn_regression_losses = []
        self._detector_class_losses = []
        self._detector_regression_losses = []

    # Memasukkan hasil loss setiap batch training
    def on_training_step(self, loss):
        self._rpn_class_losses.append(loss.rpn_class)
        self._rpn_regression_losses.append(loss.rpn_regression)
        self._detector_class_losses.append(loss.detector_class)
        self._detector_regression_losses.append(loss.detector_regression)
        self.rpn_class_loss = np.mean(self._rpn_class_losses)
        self.rpn_regression_loss = np.mean(self._rpn_regression_losses)
        self.detector_class_loss = np.mean(self._detector_class_losses)
        self.detector_regression_loss = np.mean(self._detector_regression_losses)

    # Menghasilkan format output loss untuk progress bar (tqdm)
    def get_progbar_postfix(self):
        return {
            "rpn_class_loss": "%1.4f" % self.rpn_class_loss,
            "rpn_regr_loss": "%1.4f" % self.rpn_regression_loss,
            "detector_class_loss": "%1.4f" % self.detector_class_loss,
            "detector_regr_loss": "%1.4f" % self.detector_regression_loss,
            "total_loss": "%1.2f" % (self.rpn_class_loss + self.rpn_regression_loss + self.detector_class_loss + self.detector_regression_loss)
        }

# Class untuk menghitung kurva Precision-Recall dan mAP (Mean Average Precision)
class PrecisionRecallCurveCalculator:
    def __init__(self):
        self._unsorted_predictions_by_class_index = defaultdict(list)
        self._object_count_by_class_index = defaultdict(int)

    # Menghitung akurasi prediksi per gambar (per class)
    def _compute_correctness_of_predictions(self, scored_boxes_by_class_index, gt_boxes):
        unsorted_predictions_by_class_index = {}
        object_count_by_class_index = defaultdict(int)

        for gt_box in gt_boxes:
            object_count_by_class_index[gt_box.class_index] += 1

        for class_index, scored_boxes in scored_boxes_by_class_index.items():
            gt_boxes_this_class = [gt_box for gt_box in gt_boxes if gt_box.class_index == class_index]
            ious = []
            for gt_idx in range(len(gt_boxes_this_class)):
                for box_idx in range(len(scored_boxes)):
                    boxes1 = np.expand_dims(scored_boxes[box_idx][0:4], axis=0)
                    boxes2 = np.expand_dims(gt_boxes_this_class[gt_idx].corners, axis=0)
                    iou = intersection_over_union(boxes1, boxes2)
                    ious.append((iou, box_idx, gt_idx))
            ious = sorted(ious, key=lambda iou: iou[0], reverse=True)

            gt_box_detected = [False] * len(gt_boxes)
            is_true_positive = [False] * len(scored_boxes)

            iou_threshold = 0.5
            for iou, box_idx, gt_idx in ious:
                if iou <= iou_threshold:
                    continue
                if is_true_positive[box_idx] or gt_box_detected[gt_idx]:
                    continue
                is_true_positive[box_idx] = True
                gt_box_detected[gt_idx] = True

            unsorted_predictions_by_class_index[class_index] = [
                (scored_boxes[i][4], is_true_positive[i]) for i in range(len(scored_boxes))
            ]
        return unsorted_predictions_by_class_index, object_count_by_class_index

    # Memasukkan hasil evaluasi dari 1 gambar ke kalkulasi keseluruhan
    def add_image_results(self, scored_boxes_by_class_index, gt_boxes):
        unsorted, count = self._compute_correctness_of_predictions(scored_boxes_by_class_index, gt_boxes)
        for class_index, predictions in unsorted.items():
            self._unsorted_predictions_by_class_index[class_index] += predictions
        for class_index, cnt in count.items():
            self._object_count_by_class_index[class_index] += cnt

    # Menghitung AP untuk satu kelas
    def _compute_average_precision(self, class_index):
        sorted_predictions = sorted(self._unsorted_predictions_by_class_index[class_index], key=lambda p: p[0], reverse=True)
        num_ground_truth = self._object_count_by_class_index[class_index]
        recall_array, precision_array = [], []
        tp, fp = 0, 0
        for score, correct in sorted_predictions:
            if correct:
                tp += 1
            else:
                fp += 1
            recall_array.append(tp / num_ground_truth)
            precision_array.append(tp / (tp + fp))

        recall_array.insert(0, 0.0)
        recall_array.append(1.0)
        precision_array.insert(0, 0.0)
        precision_array.append(0.0)

        for i in range(len(precision_array)):
            precision_array[i] = np.max(precision_array[i:])

        ap = 0
        for i in range(len(recall_array) - 1):
            dx = recall_array[i + 1] - recall_array[i]
            dy = precision_array[i + 1]
            ap += dy * dx
        return ap, recall_array, precision_array

    # Menghitung nilai mAP (mean Average Precision)
    def compute_mean_average_precision(self):
        average_precisions = []
        for class_index in self._object_count_by_class_index:
            ap, _, _ = self._compute_average_precision(class_index)
            average_precisions.append(ap)
        return np.mean(average_precisions)

    # Mencetak hasil AP per kelas
    def print_average_precisions(self, class_index_to_name):
        labels = [class_index_to_name[idx] for idx in self._object_count_by_class_index]
        aps = []
        for class_index in self._object_count_by_class_index:
            ap, _, _ = self._compute_average_precision(class_index)
            aps.append(ap)
        sorted_results = sorted(zip(labels, aps), key=lambda x: x[1], reverse=True)
        print("Average Precisions:\n------------------")
        for label, ap in sorted_results:
            print("%s: %1.1f%%" % (label.ljust(15), ap * 100.0))
        print("------------------")