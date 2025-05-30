#
# Faster R-CNN di PyTorch (modul utama eksekusi)
# Entry point utama untuk menjalankan training, evaluasi, dan inference Faster R-CNN.
#

import argparse
import os
import torch as t
from tqdm import tqdm

from .datasets import voc
from .models.faster_rcnn import FasterRCNNModel
from .models import vgg16, vgg16_torch, resnet
from .statistics import TrainingStatistics, PrecisionRecallCurveCalculator
from . import state, utils, visualize

# --- fungsi utama ---

def render_anchors(backbone):
  # Visualisasi anchor dari training set
  training_data = voc.Dataset(
    image_preprocessing_params=backbone.image_preprocessing_params,
    compute_feature_map_shape_fn=backbone.compute_feature_map_shape,
    feature_pixels=backbone.feature_pixels,
    dir=options.dataset_dir,
    split=options.train_split,
    augment=False,
    shuffle=False
  )
  if not os.path.exists(options.dump_anchors):
    os.makedirs(options.dump_anchors)
  print("Rendering anchors...")
  for sample in iter(training_data):
    output_path = os.path.join(options.dump_anchors, "anchors_" + os.path.basename(sample.filepath) + ".png")
    visualize.show_anchors(
      output_path=output_path,
      image=sample.image,
      anchor_map=sample.anchor_map,
      anchor_valid_map=sample.anchor_valid_map,
      gt_rpn_map=sample.gt_rpn_map,
      gt_boxes=sample.gt_boxes
    )

def evaluate(model, eval_data=None, num_samples=None, plot=False, print_average_precisions=False):
  # Evaluasi mAP
  if eval_data is None:
    eval_data = voc.Dataset(
      image_preprocessing_params=model.backbone.image_preprocessing_params,
      compute_feature_map_shape_fn=model.backbone.compute_feature_map_shape,
      feature_pixels=model.backbone.feature_pixels,
      dir=options.dataset_dir,
      split=options.eval_split,
      augment=False,
      shuffle=False
    )
  if num_samples is None:
    num_samples = eval_data.num_samples
  precision_recall_curve = PrecisionRecallCurveCalculator()
  i = 0
  print("Evaluating '%s'..." % eval_data.split)
  for sample in tqdm(iter(eval_data), total=num_samples):
    scored_boxes_by_class_index = model.predict(
      image_data=t.from_numpy(sample.image_data).unsqueeze(dim=0).cuda(),
      score_threshold=0.05
    )
    precision_recall_curve.add_image_results(
      scored_boxes_by_class_index=scored_boxes_by_class_index,
      gt_boxes=sample.gt_boxes
    )
    i += 1
    if i >= num_samples:
      break
  if print_average_precisions:
    precision_recall_curve.print_average_precisions(class_index_to_name=voc.Dataset.class_index_to_name)
  mean_average_precision = 100.0 * precision_recall_curve.compute_mean_average_precision()
  print("Mean Average Precision = %1.2f%%" % mean_average_precision)
  if plot:
    precision_recall_curve.plot_average_precisions(class_index_to_name=voc.Dataset.class_index_to_name)
  return mean_average_precision

# Membuat optimizer (SGD + weight decay)
def create_optimizer(model):
  params = []
  for key, value in dict(model.named_parameters()).items():
    if not value.requires_grad:
      continue
    if "weight" in key:
      params += [{"params": [value], "weight_decay": options.weight_decay}]
  return t.optim.SGD(params, lr=options.learning_rate, momentum=options.momentum)

# (Opsional) mengaktifkan CUDA memory profiler
def enable_cuda_memory_profiler(model):
  from pytorch.FasterRCNN import profile
  import sys
  import threading
  memory_profiler = profile.CUDAMemoryProfiler([model], filename="cuda_memory.txt")
  sys.settrace(memory_profiler)
  threading.settrace(memory_profiler)

# Fungsi utama training

def train(model):
  if options.profile_cuda_memory:
    enable_cuda_memory_profiler(model=model)

  # Inisialisasi bobot awal
  if options.load_from:
    initial_weights = options.load_from
  else:
    if options.backbone == "vgg16":
      initial_weights = "none"
    else:
      initial_weights = "IMAGENET1K_V1"

  print("Training Parameters\n-------------------")
  print("Initial weights   : %s" % initial_weights)
  print("Dataset           : %s" % options.dataset_dir)
  print("Training split    : %s" % options.train_split)
  print("Evaluation split  : %s" % options.eval_split)
  print("Backbone          : %s" % options.backbone)
  print("Epochs            : %d" % options.epochs)
  print("Learning rate     : %f" % options.learning_rate)
  print("Momentum          : %f" % options.momentum)
  print("Weight decay      : %f" % options.weight_decay)
  print("Dropout           : %f" % options.dropout)
  print("Augmentation      : %s" % ("disabled" if options.no_augment else "enabled"))
  print("Edge proposals    : %s" % ("excluded" if options.exclude_edge_proposals else "included"))
  print("CSV log           : %s" % ("none" if not options.log_csv else options.log_csv))
  print("Checkpoints       : %s" % ("disabled" if not options.checkpoint_dir else options.checkpoint_dir))

  training_data = voc.Dataset(
    dir=options.dataset_dir,
    split=options.train_split,
    image_preprocessing_params=model.backbone.image_preprocessing_params,
    compute_feature_map_shape_fn=model.backbone.compute_feature_map_shape,
    feature_pixels=model.backbone.feature_pixels,
    augment=not options.no_augment,
    shuffle=True,
    cache=options.cache_images
  )

  eval_data = voc.Dataset(
    dir=options.dataset_dir,
    split=options.eval_split,
    image_preprocessing_params=model.backbone.image_preprocessing_params,
    compute_feature_map_shape_fn=model.backbone.compute_feature_map_shape,
    feature_pixels=model.backbone.feature_pixels,
    augment=False,
    shuffle=False,
    cache=False
  )

  optimizer = create_optimizer(model=model)

  if options.checkpoint_dir and not os.path.exists(options.checkpoint_dir):
    os.makedirs(options.checkpoint_dir)

  if options.log_csv:
    csv = utils.CSVLog(options.log_csv)

  if options.save_best_to:
    best_weights_tracker = state.BestWeightsTracker(filepath=options.save_best_to)

  # Mulai training per-epoch
  for epoch in range(1, 1 + options.epochs):
    print("Epoch %d/%d" % (epoch, options.epochs))
    stats = TrainingStatistics()
    progbar = tqdm(iterable=iter(training_data), total=training_data.num_samples, postfix=stats.get_progbar_postfix())

    for sample in progbar:
      loss = model.train_step(
        optimizer=optimizer,
        image_data=t.from_numpy(sample.image_data).unsqueeze(dim=0).cuda(),
        anchor_map=sample.anchor_map,
        anchor_valid_map=sample.anchor_valid_map,
        gt_rpn_map=t.from_numpy(sample.gt_rpn_map).unsqueeze(dim=0).cuda(),
        gt_rpn_object_indices=[sample.gt_rpn_object_indices],
        gt_rpn_background_indices=[sample.gt_rpn_background_indices],
        gt_boxes=[sample.gt_boxes]
      )
      stats.on_training_step(loss=loss)
      progbar.set_postfix(stats.get_progbar_postfix())

    mean_average_precision = evaluate(
      model=model,
      eval_data=eval_data,
      num_samples=options.periodic_eval_samples,
      plot=False,
      print_average_precisions=False
    )

    # Simpan checkpoint
    if options.checkpoint_dir:
      checkpoint_file = os.path.join(options.checkpoint_dir, "checkpoint-epoch-%d-mAP-%1.1f.pth" % (epoch, mean_average_precision))
      t.save({"epoch": epoch, "model_state_dict": model.state_dict()}, checkpoint_file)
      print("Saved checkpoint to '%s'" % checkpoint_file)

    # Logging ke CSV jika diminta
    if options.log_csv:
      log_items = {
        "epoch": epoch,
        "learning_rate": options.learning_rate,
        "momentum": options.momentum,
        "weight_decay": options.weight_decay,
        "dropout": options.dropout,
        "mAP": mean_average_precision
      }
      log_items.update(stats.get_progbar_postfix())
      csv.log(log_items)

    # Simpan best model
    if options.save_best_to:
      best_weights_tracker.on_epoch_end(model=model, epoch=epoch, mAP=mean_average_precision)

  if options.save_to:
    t.save({"epoch": epoch, "model_state_dict": model.state_dict()}, options.save_to)
    print("Saved final model weights to '%s'" % options.save_to)

  if options.save_best_to:
    best_weights_tracker.save_best_weights(model=model)

  # Evaluasi model akhir
  print("Evaluating final model on full dataset...")
  evaluate(
    model=model,
    eval_data=eval_data,
    num_samples=eval_data.num_samples,
    plot=options.plot,
    print_average_precisions=True
  )

# Inference (Prediksi) --------------------------------------

def predict(model, image_data, image, show_image, output_path):
  image_data = t.from_numpy(image_data).unsqueeze(dim=0).cuda()
  scored_boxes_by_class_index = model.predict(image_data=image_data, score_threshold=0.7)
  visualize.show_detections(
    output_path=output_path,
    show_image=show_image,
    image=image,
    scored_boxes_by_class_index=scored_boxes_by_class_index,
    class_index_to_name=voc.Dataset.class_index_to_name
  )

def predict_one(model, url, show_image, output_path):
  from .datasets import image
  image_data, image_obj, _, _ = image.load_image(url=url, preprocessing=model.backbone.image_preprocessing_params, min_dimension_pixels=600)
  predict(model=model, image_data=image_data, image=image_obj, show_image=show_image, output_path=output_path)

def predict_all(model, split):
  dirname = "predictions_" + split
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  print("Rendering predictions for '%s' set..." % split)
  dataset = voc.Dataset(
    dir=options.dataset_dir,
    split=split,
    image_preprocessing_params=model.backbone.image_preprocessing_params,
    compute_feature_map_shape_fn=model.backbone.compute_feature_map_shape,
    feature_pixels=model.backbone.feature_pixels,
    augment=False,
    shuffle=False
  )
  for sample in iter(dataset):
    output_path = os.path.join(dirname, os.path.splitext(os.path.basename(sample.filepath))[0] + ".png")
    predict(model=model, image_data=sample.image_data, image=sample.image, show_image=False, output_path=output_path)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser("FasterRCNN")

  # Mode operasi utama (hanya bisa memilih satu)
  group = parser.add_mutually_exclusive_group()
  group.add_argument("--train", action="store_true", help="Melatih model")
  group.add_argument("--eval", action="store_true", help="Evaluasi model")
  group.add_argument("--predict", metavar="url", type=str, help="Inference satu gambar dan tampilkan hasilnya")
  group.add_argument("--predict-to-file", metavar="url", type=str, help="Inference satu gambar dan simpan hasil ke file")
  group.add_argument("--predict-all", metavar="name", type=str, help="Inference semua gambar dari subset dataset")

  # Parameter file dan dataset
  parser.add_argument("--load-from", metavar="file", help="Load bobot awal dari file")
  parser.add_argument("--backbone", metavar="model", default="vgg16", help="Backbone feature extractor")
  parser.add_argument("--save-to", metavar="file", help="Simpan bobot final")
  parser.add_argument("--save-best-to", metavar="file", help="Simpan bobot terbaik")
  parser.add_argument("--dataset-dir", metavar="dir", default="VOCdevkit/VOC2007", help="Direktori dataset VOC")
  parser.add_argument("--train-split", metavar="name", default="trainval", help="Split dataset untuk training")
  parser.add_argument("--eval-split", metavar="name", default="test", help="Split dataset untuk evaluasi")

  # Opsi training
  parser.add_argument("--cache-images", action="store_true", help="Cache gambar di RAM saat training")
  parser.add_argument("--periodic-eval-samples", metavar="count", default=1000, help="Jumlah sample evaluasi periodik")
  parser.add_argument("--checkpoint-dir", metavar="dir", help="Folder penyimpanan checkpoint tiap epoch")
  parser.add_argument("--plot", action="store_true", help="Plot average precision setelah evaluasi")
  parser.add_argument("--log-csv", metavar="file", help="Log training ke file CSV")

  # Hyperparameter training
  parser.add_argument("--epochs", type=int, default=1, help="Jumlah epoch")
  parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
  parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
  parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
  parser.add_argument("--dropout", type=float, default=0.0, help="Dropout setelah fully-connected detector")
  parser.add_argument("--no-augment", action="store_true", help="Nonaktifkan augmentasi gambar saat training")
  parser.add_argument("--exclude-edge-proposals", action="store_true", help="Hilangkan proposal di tepi gambar")
  parser.add_argument("--dump-anchors", metavar="dir", help="Render semua anchor object untuk analisis")
  parser.add_argument("--profile-cuda-memory", action="store_true", help="Profiling penggunaan memori CUDA")

  options = parser.parse_args()

  # Validasi backbone
  valid_backbones = ["vgg16", "vgg16-torch", "resnet50", "resnet101", "resnet152"]
  assert options.backbone in valid_backbones, "--backbone harus salah satu dari: " + ", ".join(valid_backbones)
  if options.dropout != 0:
    assert options.backbone in ["vgg16", "vgg16-torch"], "--dropout hanya berlaku untuk backbone VGG-16"

  # Load backbone sesuai pilihan user
  if options.backbone == "vgg16":
    from .models import vgg16
    backbone = vgg16.VGG16Backbone(dropout_probability=options.dropout)
  elif options.backbone == "vgg16-torch":
    from .models import vgg16_torch
    backbone = vgg16_torch.VGG16Backbone(dropout_probability=options.dropout)
  elif options.backbone == "resnet50":
    from .models import resnet
    backbone = resnet.ResNetBackbone(architecture=resnet.Architecture.ResNet50)
  elif options.backbone == "resnet101":
    from .models import resnet
    backbone = resnet.ResNetBackbone(architecture=resnet.Architecture.ResNet101)
  elif options.backbone == "resnet152":
    from .models import resnet
    backbone = resnet.ResNetBackbone(architecture=resnet.Architecture.ResNet152)

  # Jika diminta render anchor
  if options.dump_anchors:
    render_anchors(backbone=backbone)

  # Load model FasterRCNN
  from .models.faster_rcnn import FasterRCNNModel
  model = FasterRCNNModel(
    num_classes=voc.Dataset.num_classes,
    backbone=backbone,
    allow_edge_proposals=not options.exclude_edge_proposals
  ).cuda()

  if options.load_from:
    state.load(model=model, filepath=options.load_from)

  # Eksekusi sesuai opsi
  if options.train:
    train(model=model)
  elif options.eval:
    evaluate(model=model, plot=options.plot, print_average_precisions=True)
  elif options.predict:
    predict_one(model=model, url=options.predict, show_image=True, output_path=None)
  elif options.predict_to_file:
    predict_one(model=model, url=options.predict_to_file, show_image=False, output_path="predictions.png")
  elif options.predict_all:
    predict_all(model=model, split=options.predict_all)
  elif not options.dump_anchors:
    print("Tidak ada perintah yang dipilih. Gunakan --train atau --predict.")