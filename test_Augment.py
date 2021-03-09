import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.autograd import Variable
import torchvision.models as models

import itertools
from itertools import cycle
import pathlib
import time
from time import strftime
import argparse
import json

from Model import Inpaint_generator, Tumor_shape, Tumor_grade
from losses import *
from Dataloader import *
from Util import *
from pytorch_SSIM import *
from math import log10

from fid_score import calculate_fid_given_paths

parser = argparse.ArgumentParser()
parser.add_argument('--step', type=int, default=1, help='0: Tumor shape&level / 1: Tumor inpaint / 2: Whole step')
parser.add_argument('--batchSize', type=float, default=1, help='batch size for network')
parser.add_argument('--imsize', type=float, default=256, help='imsize for training data')
parser.add_argument('--dataroot', type=str, default='../data/Brain_v5', help='root of the images')

parser.add_argument('--result', type=str, default='./Result/Test', help='directory for save the result')

parser.add_argument('--model_max_num', type=int, default=110)
parser.add_argument('--resume_path_1', type=str, default='')
parser.add_argument('--resume_path_2', type=str, default='/home/sunho/PycharmProjects/ksh/venv/InpaintGAN/inpaint_result_v7/20190818_1513/model/')

opt = parser.parse_args()
print(opt)


F_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/Proposed_L1/F/', opt.dataroot +'/Train/Original_part/2/F/', 36, True, 2048)
T1_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/Proposed_L1/T1/', opt.dataroot  + '/Train/Original_part/2/T1/', 36, True, 2048)
T1c_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/Proposed_L1/T1c/', opt.dataroot  + '/Train/Original_part/2/T1c/', 36, True, 2048)
T2_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/Proposed_L1/T2/', opt.dataroot  + '/Train/Original_part/2/T2/', 36, True, 2048)
print('Proposed_pixel FID:')
print('%.04f, %.04f, %.04f, %.04f' % (F_fid_value, T1_fid_value, T1c_fid_value, T2_fid_value))

F_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/Proposed_L1_Adv/F/', opt.dataroot +'/Train/Original_part/2/F/', 36, True, 2048)
T1_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/Proposed_L1_Adv/T1/', opt.dataroot  + '/Train/Original_part/2/T1/', 36, True, 2048)
T1c_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/Proposed_L1_Adv/T1c/', opt.dataroot  + '/Train/Original_part/2/T1c/', 36, True, 2048)
T2_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/Proposed_L1_Adv/T2/', opt.dataroot  + '/Train/Original_part/2/T2/', 36, True, 2048)
print('Proposed_pixel FID:')
print('%.04f, %.04f, %.04f, %.04f' % (F_fid_value, T1_fid_value, T1c_fid_value, T2_fid_value))
F_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/Proposed_L1_Adv_Cont/F/', opt.dataroot +'/Train/Original_part/2/F/', 36, True, 2048)
T1_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/Proposed_L1_Adv_Cont/T1/', opt.dataroot  + '/Train/Original_part/2/T1/', 36, True, 2048)
T1c_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/Proposed_L1_Adv_Cont/T1c/', opt.dataroot  + '/Train/Original_part/2/T1c/', 36, True, 2048)
T2_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/Proposed_L1_Adv_Cont/T2/', opt.dataroot  + '/Train/Original_part/2/T2/', 36, True, 2048)
print('Proposed_pixel FID:')
print('%.04f, %.04f, %.04f, %.04f' % (F_fid_value, T1_fid_value, T1c_fid_value, T2_fid_value))
F_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/CBGAN/F/', opt.dataroot +'/Train/Original_part/2/F/', 36, True, 2048)
T1_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/CBGAN/T1/', opt.dataroot  + '/Train/Original_part/2/T1/', 36, True, 2048)
T1c_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/CBGAN/T1c/', opt.dataroot  + '/Train/Original_part/2/T1c/', 36, True, 2048)
T2_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/CBGAN/T2/', opt.dataroot  + '/Train/Original_part/2/T2/', 36, True, 2048)
print('Proposed_pixel FID:')
print('%.04f, %.04f, %.04f, %.04f' % (F_fid_value, T1_fid_value, T1c_fid_value, T2_fid_value))
F_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/Pix2pix/F/', opt.dataroot +'/Train/Original_part/2/F/', 36, True, 2048)
T1_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/Pix2pix/T1/', opt.dataroot  + '/Train/Original_part/2/T1/', 36, True, 2048)
T1c_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/Pix2pix/T1c/', opt.dataroot  + '/Train/Original_part/2/T1c/', 36, True, 2048)
T2_fid_value = calculate_fid_given_paths('/home/sunho/PycharmProjects/ksh/venv/Figure/FID/Pix2pix/T2/', opt.dataroot  + '/Train/Original_part/2/T2/', 36, True, 2048)
print('Proposed_pixel FID:')
print('%.04f, %.04f, %.04f, %.04f' % (F_fid_value, T1_fid_value, T1c_fid_value, T2_fid_value))
