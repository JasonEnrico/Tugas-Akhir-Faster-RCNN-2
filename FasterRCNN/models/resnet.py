from math import ceil
import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision

from ..datasets import image
from .backbone import Backbone

class FeatureExtractor(nn.Module):
  def __init__(self, resnet):
    super().__init__()

    # Layer ekstraksi fitur
    self._feature_extractor = nn.Sequential(
      resnet.conv1,     # 0
      resnet.bn1,       # 1
      resnet.relu,      # 2
      resnet.maxpool,   # 3
      resnet.layer1,    # 4
      resnet.layer2,    # 5
      resnet.layer3     # 6
    )

    # Membekukan (freeze) layer awal agar bobot tidak dilatih ulang
    self._freeze(resnet.conv1)
    self._freeze(resnet.bn1)
    self._freeze(resnet.layer1)

    # Membekukan seluruh layer batch normalization seperti dijelaskan pada Appendix A di paper asli ResNet
    self._freeze_batchnorm(self._feature_extractor)

  # Override fungsi train() bawaan nn.Module
  def train(self, mode = True):
    super().train(mode)

    if mode:
      # Saat training, pastikan blok tertentu tetap dalam mode evaluasi
      self._feature_extractor.eval()
      self._feature_extractor[5].train()
      self._feature_extractor[6].train()

      # Seluruh layer batchnorm tetap dalam mode evaluasi
      def set_bn_eval(module):
        if type(module) == nn.BatchNorm2d:
          module.eval()
      self._feature_extractor.apply(set_bn_eval)

  def forward(self, image_data):
    y = self._feature_extractor(image_data)
    return y

  @staticmethod
  def _freeze(layer):
    for name, parameter in layer.named_parameters():
      parameter.requires_grad = False

  def _freeze_batchnorm(self, block):
    for child in block.modules():
      if type(child) == nn.BatchNorm2d:
        self._freeze(layer = child)

class PoolToFeatureVector(nn.Module):
  def __init__(self, resnet):
    super().__init__()
    self._layer4 = resnet.layer4
    self._freeze_batchnorm(self._layer4)

  def train(self, mode = True):
    super().train(mode)
    if mode:
      def set_bn_eval(module):
        if type(module) == nn.BatchNorm2d:
          module.eval()
      self._layer4.apply(set_bn_eval)

  def forward(self, rois):
    y = self._layer4(rois)  # (N, 1024, 7, 7) -> (N, 2048, 4, 4)

    # Rata-ratakan dua dimensi terakhir -> (N, 2048)
    y = y.mean(-1).mean(-1)
    return y

  @staticmethod
  def _freeze(layer):
    for name, parameter in layer.named_parameters():
      parameter.requires_grad = False

  def _freeze_batchnorm(self, block):
    for child in block.modules():
      if type(child) == nn.BatchNorm2d:
        self._freeze(layer = child)

class ResNetBackbone(Backbone):
  def __init__(self):
    super().__init__()

    # Properti backbone:
    self.feature_map_channels = 1024  # jumlah channel keluaran dari feature extractor
    self.feature_pixels = 16          # resolusi feature map adalah 1/16 dari ukuran gambar asli
    self.feature_vector_size = 2048   # ukuran vektor fitur setelah pooling
    self.image_preprocessing_params = image.PreprocessingParams(
      channel_order = image.ChannelOrder.RGB,
      scaling = 1.0 / 255.0,
      means = [ 0.485, 0.456, 0.406 ],
      stds = [ 0.229, 0.224, 0.225 ]
    )

    # Membuat model dan memuat bobot ImageNet (pre-trained)
    resnet = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    print("Bobot pra-latih IMAGENET1K_V1 untuk backbone Torchvision ResNet50 berhasil dimuat")

    # Feature extractor: menerima input (batch_size, channels, height, width),
    # menghasilkan feature map (batch_size, 1024, ceil(height/16), ceil(width/16))
    self.feature_extractor = FeatureExtractor(resnet = resnet)

    # Mengubah hasil pooling menjadi feature vector untuk head
    self.pool_to_feature_vector = PoolToFeatureVector(resnet = resnet)

  def compute_feature_map_shape(self, image_shape):
    """
    Menghitung bentuk feature map berdasarkan bentuk gambar input.

    Parameter
    ---------
    image_shape : Tuple[int, int, int]
      Bentuk input gambar (channels, height, width).

    Return
    ------
    Tuple[int, int, int]
      Bentuk feature map: (feature_map_channels, feature_map_height, feature_map_width)
    """
    image_width = image_shape[-1]
    image_height = image_shape[-2]
    return (self.feature_map_channels, ceil(image_height / self.feature_pixels), ceil(image_width / self.feature_pixels))
