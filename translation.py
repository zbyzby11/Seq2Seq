"""
进行翻译的主文件
"""
import torch
from torch import nn, optim
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq

class Translation(object):
    def __init__(self):
        ...