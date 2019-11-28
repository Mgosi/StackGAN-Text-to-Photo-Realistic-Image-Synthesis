import os.path as osp
import numpy as np
from easydict import EasyDict as edict

var = edict()
config = var

#Model Values
var.TEXT = edict()
var.TEXT.DIMENSION = 1024

var.GAN = edict()
var.GAN.CONDITION_DIM = 128
var.GAN.DF_DIM = 64
var.GAN.GF_DIM = 128
var.GAN.RES_NUM = 4

#Training Values
var.TRAIN = edict()
var.TRAIN.BATCH_SIZE = 64
var.TRAIN.MAX_EPOCH = 500
var.TRAIN.DIS_LR = 0.0002
var.TRAIN.GEN_LR = 0.0002

var.Z_DIM = 100
var.IMG_SIZE = 64
var.CUDA = False
var.DEVICE = 'cpu'