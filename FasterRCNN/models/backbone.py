#
# Faster R-CNN di PyTorch (khusus ResNet50 backbone)
# Backbone dasar untuk feature extractor dan pengubah hasil pooling menjadi vektor fitur.
# Digunakan di tahap:
#   1. Extractor fitur awal (input gambar → feature map → RPN dan detektor)
#   2. Pooling hasil proposal → feature vector → klasifikasi dan regresi box
#


import torch as t
from torch import nn
from ..datasets import image

class Backbone:
  """
  Kelas dasar untuk backbone. Semua backbone (seperti ResNet50) akan mewarisi kelas ini.
  """
  def __init__(self):
    # Properti utama backbone:
    self.feature_map_channels = 0    # jumlah channel output feature map
    self.feature_pixels = 0          # ukuran piksel per sel feature map (misal: 16 berarti 1 sel merepresentasikan 16x16 piksel citra asli)
    self.feature_vector_size = 0     # ukuran vektor fitur linear setelah pooling, sebelum masuk ke detektor
    
    # Parameter preprocessing gambar (default BGR, biasanya digunakan pada VGG)
    self.image_preprocessing_params = image.PreprocessingParams(
      channel_order = image.ChannelOrder.BGR,
      scaling = 1.0,
      means = [103.939, 116.779, 123.680],
      stds = [1, 1, 1]
    )

    # Komponen utama backbone:
    self.feature_extractor = None      # nn.Module untuk mengekstrak feature map dari gambar input
    self.pool_to_feature_vector = None # nn.Module untuk mengubah RoI hasil pooling menjadi vektor fitur linear

  def compute_feature_map_shape(self, image_shape):
    """
    Menghitung bentuk feature map hasil dari feature extractor berdasarkan ukuran gambar input.
    
    Parameter
    ---------
    image_shape : Tuple[int, int, int]
      Ukuran gambar input (channels, height, width).

    Return
    ------
    Tuple[int, int, int]
      Bentuk feature map: (feature_map_channels, feature_map_height, feature_map_width)
    """
    return image_shape[-3:]
