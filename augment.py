from fid_score import calculate_fid_given_paths
from util import *
from model import Inpaint_generator, Tumor_shape, Tumor_grade
import shutil
import math
import random
from math import log10
from dataloader import *
from losses import *
import json
import argparse
from time import strftime
import time
import pathlib
from itertools import cycle
import itertools
import torchvision.transforms.functional as TF
import torchvision.models as models
from torch.autograd import Variable
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
parser = argparse.ArgumentParser()
parser.add_argument('--step', type=int, default=1,
                    help='0: Tumor shape&level / 1: Tumor inpaint / 2: Whole step')
parser.add_argument('--dataroot', type=str,
                    default='../data/Brain_v5', help='root of the images')
parser.add_argument('--temp_dataroot', type=str,
                    default='/home/sunho/PycharmProjects/ksh/venv/data/Brain_v5/Train/Normal_slice/', help='root of the images')
parser.add_argument('--Augment_dir', type=str,
                    default='/home/sunho/PycharmProjects/ksh/venv/data/Brain_v5/Train/CNN_Augment/Using_Normal_slice/L1_Adv_Cont/', help='directory for save the result')


# parser.add_argument('--resume_path_1', type=str, default='/home/sunho/PycharmProjects/ksh/venv/InpaintGAN/Result/STEP0/model/190/')
parser.add_argument('--resume_path_1', type=str,
                    default='/home/sunho/PycharmProjects/ksh/venv/InpaintGAN/inpaint_result_v7/20190815_1644/model/40/')
parser.add_argument('--resume_path_2', type=str,
                    default='/home/sunho/PycharmProjects/ksh/venv/InpaintGAN/Result/L1/model/100/')

opt = parser.parse_args()
print(opt)
if pathlib.Path('%s' % (opt.dataroot)).exists() == False:
    print("******  Data root is incorrect...******")
Tensor = torch.cuda.FloatTensor

Augment_dir = opt.Augment_dir
model_dir1 = opt.resume_path_1
model_dir2 = opt.resume_path_2


tumor_shape = Tumor_shape().cuda()
tumor_grade = Tumor_grade().cuda()

inp_gen_L1 = Inpaint_generator().cuda()
inp_gen_L1_Cont = Inpaint_generator().cuda()
inp_gen_L1_Adv = Inpaint_generator().cuda()
inp_gen_L1_Cont_Adv = Inpaint_generator().cuda()

# inp_gen = UNet(5,4).cuda()
tumor_shape.load_state_dict(torch.load(model_dir1 + 'tumor_shape.pth'))
tumor_grade.load_state_dict(torch.load(
    '/home/sunho/PycharmProjects/ksh/venv/InpaintGAN/Result/Grade/model/190/tumor_grade.pth'))

inp_gen_L1.load_state_dict(torch.load(
    '/home/sunho/PycharmProjects/ksh/venv/InpaintGAN/Result/L1/model/100/' + 'inp_gen.pth'))
inp_gen_L1_Cont.load_state_dict(torch.load(
    '/home/sunho/PycharmProjects/ksh/venv/InpaintGAN/Result/20190826_2005/model/40/' + 'inp_gen.pth'))
inp_gen_L1_Adv.load_state_dict(torch.load(
    '/home/sunho/PycharmProjects/ksh/venv/InpaintGAN/Result/L1_ADV/model/100/' + 'inp_gen.pth'))
inp_gen_L1_Cont_Adv.load_state_dict(torch.load(
    '/home/sunho/PycharmProjects/ksh/venv/InpaintGAN/Result/20190821_1643/model/40/' + 'inp_gen.pth'))

transforms_ = [transforms.Resize(256, 256),
               transforms.ToTensor()]
temp_augment_dataloader = DataLoader(an_BrainDataset(
    root=opt.temp_dataroot, mode='Augment', transforms_=transforms_), batch_size=1, shuffle=True, num_workers=1)

with torch.no_grad():
    aug_dir = '/home/sunho/PycharmProjects/ksh/venv/Figure/'
    j = 0
    flag = 0
    step = 3
    for epoch in range(25):
        for k, (batch) in enumerate(temp_augment_dataloader, 0):

            F, T1, T1c, T2, uni_B, M, Binary_M, Circle_M, level_Circle_M = batch['F'].type(Tensor), batch[
                'T1'].type(Tensor), batch['T1c'].type(Tensor), batch['T2'].type(Tensor), batch['uni_Brain'].type(Tensor), batch['M'].type(Tensor), batch['Binary_M'].type(Tensor), batch['Circle'].type(Tensor), batch[
                'level_Circle'].type(Tensor)
            brain = torch.cat((F, T1, T1c, T2), 1)

            # STEP 1: GENERATE VARIOUS BINARY MASK********************************************************************************************
            if (step == 1) and (j < 100):

                pathlib.Path(aug_dir + "STEP1/%d/inputs/" %
                             (j)).mkdir(parents=True, exist_ok=True)
                pathlib.Path(aug_dir + "STEP1/%d/outputs/" %
                             (j)).mkdir(parents=True, exist_ok=True)

                ran_radius = random.sample(range(1, 40), 3)
                ran_radius.sort()
                [radius3, radius2, radius1] = ran_radius

                # ran_center = random.sample(range(80, 160), 2)
                # [x_center, y_center] = ran_center

                for i in range(25):

                    level_circle = make_level_circle(
                        128 + 26*math.cos((2*math.pi/25)*i), 128 + 26*math.sin((2*math.pi/25)*i), radius1, radius2, radius3)

                    binary_circle = unify(level_circle).cuda()
                    binary_mask = unify(tumor_shape(
                        torch.cat([uni_B, binary_circle], 1)).cpu()).cuda()

                    if i == 0:
                        inputs = binary_circle + uni_B*0.3
                        outputs = binary_mask
                    inputs = torch.cat((inputs, binary_circle + uni_B*0.3), 0)
                    outputs = torch.cat((outputs, binary_mask), 0)

                    vutils.save_image(
                        binary_circle + uni_B*0.3, filename=aug_dir + 'STEP1/%d/inputs/%d.png' % (j, i), nrow=1)
                    vutils.save_image(
                        binary_mask, filename=aug_dir + 'STEP1/%d/outputs/%d.png' % (j, i), nrow=1)

                vutils.save_image(inputs, filename=aug_dir +
                                  'STEP1/%d_inputs_.png' % (j), nrow=4)
                vutils.save_image(outputs, filename=aug_dir +
                                  'STEP1/%d_outputs_.png' % (j), nrow=4)

                j = j+1

            elif (step == 2) and (j < 100):
                aug_dir = '/home/sunho/PycharmProjects/ksh/venv/Figure/Fig6/'

                pathlib.Path(aug_dir + "_STEP2/%d/inputs/" %
                             (j)).mkdir(parents=True, exist_ok=True)
                pathlib.Path(aug_dir + "_STEP2/%d/outputs/" %
                             (j)).mkdir(parents=True, exist_ok=True)

                ran_center = random.sample(range(60, 190), 2)
                [x_center, y_center] = ran_center

                ran_max = random.sample(range(3, 40), 1)
                [radius1] = ran_max

                level_circle = (make_level_circle(
                    x_center, y_center, radius1, 2, 1)).cuda()

                binary_circle = unify(level_circle.cpu()).cuda()
                binary_mask = unify(tumor_shape(
                    torch.cat([uni_B, binary_circle], 1)).cpu()).cuda()

                ran_radius_first = random.sample(range(1, radius1), 2)
                ran_radius_first.sort()
                [radius3_first, radius2_first] = ran_radius_first

                for i in range(50):
                    if i % 10 == 0:
                        if i != 0:
                            [radius3_first, radius2_first] = [
                                radius3_second, radius2_second]
                        idx = 0
                        ran_radius_second = random.sample(range(1, radius1), 2)
                        ran_radius_second.sort()
                        [radius3_second, radius2_second] = ran_radius_second

                    level_circle = (make_level_circle(x_center, y_center, radius1, (radius2_first*(
                        10-idx)+radius2_second*(idx))/10, (radius3_first*(10-idx)+radius3_second*(idx))/10)).cuda()

                    grade_mask = qauntize(tumor_grade(
                        torch.cat([binary_mask, level_circle.cuda()], 1)).cpu()).cuda()
                    idx = idx+1

                    print(idx, radius1, (radius2_first*(10-idx)+radius2_second *
                                         (idx))/10, (radius3_first*(10-idx)+radius3_second*(idx))/10)

                    if i == 0:
                        inputs = level_circle
                        outputs = grade_mask
                    inputs = torch.cat((inputs, level_circle), 0)
                    outputs = torch.cat((outputs, grade_mask), 0)

                    vutils.save_image(
                        level_circle, filename=aug_dir + '_STEP2/%d/inputs/%d.png' % (j, i), nrow=4)
                    vutils.save_image(
                        grade_mask, filename=aug_dir + '_STEP2/%d/outputs/%d.png' % (j, i), nrow=4)

                if j < 100:
                    vutils.save_image(
                        inputs, filename=aug_dir + '_STEP2/%d_inputs_.png' % (j), nrow=5)
                    vutils.save_image(
                        outputs, filename=aug_dir + '_STEP2/%d_outputs_.png' % (j), nrow=5)

                    j = j+1
                    print(j)

            elif step == 3:
                if k == 0:
                    level_circle1 = (make_level_circle(
                        100, 100, 10, 20, 30)).cuda()

                if j < 50:
                    aug_dir = '/home/sunho/PycharmProjects/ksh/venv/Figure/Fig8_v2/'

                    level_circle1 = (make_level_circle(
                        100, 120, 20, 27, 30)).cuda()
                    level_circle2 = (make_level_circle(
                        100, 120, 12, 15, 18)).cuda()
                    level_circle3 = (make_level_circle(
                        100, 120, 10, 13, 30)).cuda()
                    level_circle4 = (make_level_circle(
                        150, 150, 20, 25, 30)).cuda()

                    binary_circle1 = unify(level_circle1.cpu()).cuda()
                    binary_circle2 = unify(level_circle2.cpu()).cuda()
                    binary_circle3 = unify(level_circle3.cpu()).cuda()
                    binary_circle4 = unify(level_circle4.cpu()).cuda()

                    binary_mask1 = unify(tumor_shape(
                        torch.cat([uni_B, binary_circle1], 1)).cpu()).cuda()
                    binary_mask2 = unify(tumor_shape(
                        torch.cat([uni_B, binary_circle2], 1)).cpu()).cuda()
                    binary_mask3 = unify(tumor_shape(
                        torch.cat([uni_B, binary_circle3], 1)).cpu()).cuda()
                    binary_mask4 = unify(tumor_shape(
                        torch.cat([uni_B, binary_circle4], 1)).cpu()).cuda()

                    grade_mask1 = qauntize(tumor_grade(
                        torch.cat([binary_mask1, level_circle1.cuda()], 1)).cpu()).cuda()
                    grade_mask2 = qauntize(tumor_grade(
                        torch.cat([binary_mask2, level_circle2.cuda()], 1)).cpu()).cuda()
                    grade_mask3 = qauntize(tumor_grade(
                        torch.cat([binary_mask3, level_circle3.cuda()], 1)).cpu()).cuda()
                    grade_mask4 = qauntize(tumor_grade(
                        torch.cat([binary_mask4, level_circle4.cuda()], 1)).cpu()).cuda()

                    # if (torch.sum(binary_mask1 * (1 - uni_B)) == 0):
                    if (torch.sum(binary_mask1 * (1 - uni_B)) == 0) & (torch.sum(binary_mask2 * (1 - uni_B)) == 0) & (torch.sum(binary_mask3 * (1 - uni_B)) == 0) & (torch.sum(binary_mask4 * (1 - uni_B)) == 0):
                        vutils.save_image(
                            F, filename=aug_dir + "/F_%04d_0.png" % (j))
                        vutils.save_image(
                            T1, filename=aug_dir + "/T1_%04d_0.png" % (j))
                        vutils.save_image(
                            T1c, filename=aug_dir + "/T1c_%04d_0.png" % (j))
                        vutils.save_image(
                            T2, filename=aug_dir + "/T2_%04d_0.png" % (j))

                        brain_blank = brain * (1 - binary_mask1)
                        out_brain = inp_gen_L1_Cont_Adv(
                            brain_blank, grade_mask1)
                        F, T1, T1c, T2 = torch.split(
                            out_brain, split_size_or_sections=1, dim=1)
                        vutils.save_image(
                            F, filename=aug_dir + "/F_%04d_1.png" % (j))
                        vutils.save_image(
                            T1, filename=aug_dir + "/T1_%04d_1.png" % (j))
                        vutils.save_image(
                            T1c, filename=aug_dir + "/T1c_%04d_1.png" % (j))
                        vutils.save_image(
                            T2, filename=aug_dir + "/T2_%04d_1.png" % (j))
                        vutils.save_image(
                            grade_mask1, filename=aug_dir + "/M_%04d_1.png" % (j))
                        vutils.save_image(
                            level_circle1, filename=aug_dir + "/C_%04d_1.png" % (j))

                        brain_blank = brain * (1 - binary_mask2)
                        out_brain = inp_gen_L1_Cont_Adv(
                            brain_blank, grade_mask2)
                        F, T1, T1c, T2 = torch.split(
                            out_brain, split_size_or_sections=1, dim=1)
                        vutils.save_image(
                            F, filename=aug_dir + "/F_%04d_2.png" % (j))
                        vutils.save_image(
                            T1, filename=aug_dir + "/T1_%04d_2.png" % (j))
                        vutils.save_image(
                            T1c, filename=aug_dir + "/T1c_%04d_2.png" % (j))
                        vutils.save_image(
                            T2, filename=aug_dir + "/T2_%04d_2.png" % (j))
                        vutils.save_image(
                            grade_mask2, filename=aug_dir + "/M_%04d_2.png" % (j))
                        vutils.save_image(
                            level_circle2, filename=aug_dir + "/C_%04d_2.png" % (j))

                        brain_blank = brain * (1 - binary_mask3)
                        out_brain = inp_gen_L1_Cont_Adv(
                            brain_blank, grade_mask3)
                        F, T1, T1c, T2 = torch.split(
                            out_brain, split_size_or_sections=1, dim=1)
                        vutils.save_image(
                            F, filename=aug_dir + "/F_%04d_3.png" % (j))
                        vutils.save_image(
                            T1, filename=aug_dir + "/T1_%04d_3.png" % (j))
                        vutils.save_image(
                            T1c, filename=aug_dir + "/T1c_%04d_3.png" % (j))
                        vutils.save_image(
                            T2, filename=aug_dir + "/T2_%04d_3.png" % (j))
                        vutils.save_image(
                            grade_mask3, filename=aug_dir + "/M_%04d_3.png" % (j))
                        vutils.save_image(
                            level_circle3, filename=aug_dir + "/C_%04d_3.png" % (j))

                        brain_blank = brain * (1 - binary_mask4)
                        out_brain = inp_gen_L1_Cont_Adv(
                            brain_blank, grade_mask4)
                        F, T1, T1c, T2 = torch.split(
                            out_brain, split_size_or_sections=1, dim=1)
                        vutils.save_image(
                            F, filename=aug_dir + "/F_%04d_4.png" % (j))
                        vutils.save_image(
                            T1, filename=aug_dir + "/T1_%04d_4.png" % (j))
                        vutils.save_image(
                            T1c, filename=aug_dir + "/T1c_%04d_4.png" % (j))
                        vutils.save_image(
                            T2, filename=aug_dir + "/T2_%04d_4.png" % (j))
                        vutils.save_image(
                            grade_mask4, filename=aug_dir + "/M_%04d_4.png" % (j))
                        vutils.save_image(
                            level_circle4, filename=aug_dir + "/C_%04d_4.png" % (j))

                        j = j+1
                        print(j)
