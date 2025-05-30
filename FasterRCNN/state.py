#
# Loader dan Saver Model Weights (State Management) untuk Faster R-CNN
# Bertugas memuat dan menyimpan bobot model dari berbagai sumber: PyTorch, Caffe, Keras
#

import h5py
import numpy as np
import torch as t

# Fungsi load bobot layer dari file Keras (HDF5)
def _load_keras_weights(hdf5_file, layer_name):
    primary_keypath = "model_weights/" + layer_name
    for keypath, node in hdf5_file[primary_keypath].items():
        if keypath.startswith("conv") or keypath.startswith("dense"):
            kernel_keypath = "/".join([primary_keypath, keypath, "kernel:0"])
            weights = np.array(hdf5_file[kernel_keypath]).astype(np.float32)
            return t.from_numpy(weights).cuda()
    return None

# Fungsi load bias dari file Keras
def _load_keras_biases(hdf5_file, layer_name):
    primary_keypath = "model_weights/" + layer_name
    for keypath, node in hdf5_file[primary_keypath].items():
        if keypath.startswith("conv") or keypath.startswith("dense"):
            bias_keypath = "/".join([primary_keypath, keypath, "bias:0"])
            biases = np.array(hdf5_file[bias_keypath]).astype(np.float32)
            return t.from_numpy(biases).cuda()
    return None

# Menggabungkan load bobot dan bias sekaligus
def _load_keras_layer(hdf5_file, layer_name):
    return _load_keras_weights(hdf5_file, layer_name), _load_keras_biases(hdf5_file, layer_name)

# Khusus untuk load layer Conv2D dari Keras (format kernel Keras berbeda dengan PyTorch)
def _load_keras_conv2d_layer(hdf5_file, layer_name, keras_shape=None):
    weights, biases = _load_keras_layer(hdf5_file, layer_name)
    if weights is not None and biases is not None:
        if keras_shape is not None:
            weights = weights.reshape(keras_shape)
        weights = weights.permute([3, 2, 0, 1])  # ubah ke format PyTorch
    return weights, biases

# Load model VGG16 yang sebelumnya di-train di Keras (versi penulis kode)
def _load_vgg16_from_bart_keras_model(filepath):
    missing_layers = []
    state = {}
    file = h5py.File(filepath, "r")

    keras_layers = [
        "block1_conv1", "block1_conv2", "block2_conv1", "block2_conv2",
        "block3_conv1", "block3_conv2", "block3_conv3",
        "block4_conv1", "block4_conv2", "block4_conv3",
        "block5_conv1", "block5_conv2", "block5_conv3"
    ]
    for layer_name in keras_layers:
        weights, biases = _load_keras_conv2d_layer(file, layer_name)
        if weights is not None and biases is not None:
            state["_stage1_feature_extractor._" + layer_name + ".weight"] = weights
            state["_stage1_feature_extractor._" + layer_name + ".bias"] = biases
        else:
            missing_layers.append(layer_name)

    # Khusus untuk detector (FC layer di Keras -> FC PyTorch)
    weights, biases = _load_keras_layer(file, "fc1")
    if weights is not None and biases is not None:
        weights = weights.reshape((7, 7, 512, 4096)).permute([2, 0, 1, 3]).reshape((-1, 4096)).permute([1, 0])
        state["_stage3_detector_network._fc1.weight"] = weights
        state["_stage3_detector_network._fc1.bias"] = biases
    else:
        missing_layers.append("fc1")

    weights, biases = _load_keras_layer(file, "fc2")
    if weights is not None and biases is not None:
        state["_stage3_detector_network._fc2.weight"] = weights.permute([1, 0])
        state["_stage3_detector_network._fc2.bias"] = biases
    else:
        missing_layers.append("fc2")

    if missing_layers:
        print("Beberapa layer Keras tidak ditemukan dan tidak dimuat:", ", ".join(missing_layers))
    return state

# Load model VGG16 dari pretrained Caffe model
# Banyak dipakai karena model pretrained Caffe sering digunakan di Faster R-CNN

def _load_vgg16_from_caffe_model(filepath):
    state = {}
    caffe = t.load(filepath)
    mapping = {
        "features.0.": "_stage1_feature_extractor._block1_conv1",
        "features.2.": "_stage1_feature_extractor._block1_conv2",
        "features.5.": "_stage1_feature_extractor._block2_conv1",
        "features.7.": "_stage1_feature_extractor._block2_conv2",
        "features.10.": "_stage1_feature_extractor._block3_conv1",
        "features.12.": "_stage1_feature_extractor._block3_conv2",
        "features.14.": "_stage1_feature_extractor._block3_conv3",
        "features.17.": "_stage1_feature_extractor._block4_conv1",
        "features.19.": "_stage1_feature_extractor._block4_conv2",
        "features.21.": "_stage1_feature_extractor._block4_conv3",
        "features.24.": "_stage1_feature_extractor._block5_conv1",
        "features.26.": "_stage1_feature_extractor._block5_conv2",
        "features.28.": "_stage1_feature_extractor._block5_conv3",
        "classifier.0.": "_stage3_detector_network._fc1",
        "classifier.3.": "_stage3_detector_network._fc2"
    }
    missing_layers = set([k[:-1] for k in mapping.keys()])
    for key, tensor in caffe.items():
        caffe_layer = ".".join(key.split(".")[:2]) + "."
        if caffe_layer in mapping:
            if caffe_layer + "weight" in caffe and caffe_layer + "bias" in caffe:
                state[mapping[caffe_layer] + ".weight"] = caffe[caffe_layer + "weight"]
                state[mapping[caffe_layer] + ".bias"] = caffe[caffe_layer + "bias"]
                missing_layers.discard(caffe_layer[:-1])

    if len(missing_layers) == len(mapping):
        raise ValueError("File '%s' bukan Caffe VGG-16 model valid" % filepath)
    if missing_layers:
        print("Beberapa layer dari Caffe tidak ditemukan:", ", ".join(missing_layers))
    return state

# Loader utama model state dari file
# Bisa baca file PyTorch state, Keras model, atau Caffe model

def load(model, filepath):
    state = None
    try:
        state = _load_vgg16_from_bart_keras_model(filepath)
        print("Berhasil load dari model Keras:", filepath)
    except:
        pass
    if state is None:
        try:
            state = _load_vgg16_from_caffe_model(filepath)
            print("Berhasil load dari model Caffe:", filepath)
        except:
            pass
    if state is None:
        state = t.load(filepath)
        if "model_state_dict" not in state:
            raise KeyError("File tidak memuat 'model_state_dict': %s" % filepath)
        state = state["model_state_dict"]
    try:
        model.load_state_dict(state)
        print("Berhasil memuat bobot dari:", filepath)
    except Exception as e:
        print(e)
        return

# Class tracker untuk menyimpan bobot terbaik selama training berdasarkan mAP
class BestWeightsTracker:
    def __init__(self, filepath):
        self._filepath = filepath
        self._best_state = None
        self._best_mAP = 0

    def on_epoch_end(self, model, epoch, mAP):
        if mAP > self._best_mAP:
            self._best_mAP = mAP
            self._best_state = {"epoch": epoch, "model_state_dict": model.state_dict()}

    def save_best_weights(self, model):
        if self._best_state is not None:
            t.save(self._best_state, self._filepath)
            print("Bobot terbaik (mAP=%1.2f%%) disimpan ke %s" % (self._best_mAP, self._filepath))
