import timeit
from util import *
from dataloader import *
from losses import *
from model import Inpaint_generator, Inpaint_discriminator, Tumor_shape, Tumor_grade, Vgg16
import json
import argparse
from time import strftime
import time
import pathlib
from itertools import cycle
import itertools
import torchvision.models as models
from torch.autograd import Variable
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


parser = argparse.ArgumentParser()
parser.add_argument('--step', type=int, default=1,
                    help='0: Tumor shape & grade / 1: Tumor inpaint / 2: Whole step')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate for encoder')
parser.add_argument('--batchSize', type=float, default=1,
                    help='batch size for network')
parser.add_argument('--imsize', type=float, default=256,
                    help='imsize for training data')
parser.add_argument('--start_epoch', type=float, default=0, help='start Epoch')
parser.add_argument('--end_epoch', type=float, default=200, help='end Epoch')
parser.add_argument('--data_root', type=str,
                    default='../data/', help='root of the images')
parser.add_argument('--result', type=str,
                    default='./RESULT/')
parser.add_argument('--resume', type=str,
                    default='./MODEL/')
opt = parser.parse_args()

if pathlib.Path('%s' % (opt.data_root)).exists() == False:
    print("******  Data root is incorrect...******")

if pathlib.Path(opt.result).exists() == False:
    pathlib.Path(opt.result).mkdir(parents=True, exist_ok=True)

cudaAvailable = True
Tensor = torch.cuda.FloatTensor

'''
step == 0: only train tumor_shape network and tumor_grade network
step == 1: only train Inpaint_generator network and Inpaint_discriminator network
step == 2: train whole network
'''
if opt.step == 0:
    tumor_shape = Tumor_shape().cuda()
    tumor_grade = Tumor_grade().cuda()
elif opt.step == 1:
    inp_gen = Inpaint_generator().cuda()
    inp_dis = Inpaint_discriminator().cuda()
elif opt.step == 2:
    tumor_shape = Tumor_shape().cuda()
    tumor_grade = Tumor_grade().cuda()
    inp_gen = Inpaint_generator().cuda()
    inp_dis = Inpaint_discriminator().cuda()

vgg = Vgg16(requires_grad=False).cuda()
vgg.eval()

if opt.resume != '':
    model_dir = opt.resume
    if opt.step == 0:
        tumor_shape.load_state_dict(torch.load(model_dir+'/tumor_shape.pth'))
        tumor_grade.load_state_dict(torch.load(model_dir + '/tumor_grade.pth'))
    elif opt.step == 1:
        inp_gen.load_state_dict(torch.load(model_dir + '/inp_gen.pth'))
        inp_dis.load_state_dict(torch.load(model_dir + '/inp_dis.pth'))
    else:
        tumor_shape.load_state_dict(torch.load(model_dir + '/tumor_shape.pth'))
        tumor_grade.load_state_dict(torch.load(model_dir + '/tumor_grade.pth'))
        inp_gen.load_state_dict(torch.load(model_dir + '/inp_gen.pth'))
        inp_dis.load_state_dict(torch.load(model_dir + '/inp_dis.pth'))

if opt.step == 0:
    optm_tm_shape = torch.optim.Adam(
        tumor_shape.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optm_tm_grade = torch.optim.Adam(
        tumor_grade.parameters(), lr=opt.lr, betas=(0.5, 0.999))
elif opt.step == 1:
    optm_inp_gen = torch.optim.Adam(
        inp_gen.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optm_inp_dis = torch.optim.Adam(
        inp_dis.parameters(), lr=opt.lr, betas=(0.5, 0.999))
else:
    optm_tm_shape = torch.optim.Adam(
        tumor_shape.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optm_tm_grade = torch.optim.Adam(
        tumor_grade.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optm_inp_gen = torch.optim.Adam(
        inp_gen.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optm_inp_dis = torch.optim.Adam(
        inp_dis.parameters(), lr=opt.lr, betas=(0.5, 0.999))


# transforms_ = [transforms.Resize(256, 256), transforms.ToTensor()]
# an_dataloader = DataLoader(an_BrainDataset(root=opt.dataroot, mode='Train/Original_full/', transforms_=transforms_),batch_size=opt.batchSize, shuffle=True, num_workers=1)

abnormal_dataset = an_BrainDataset(
    root=opt.data_root, mode='Train', data_dir='/Original_full/', aug=None)
abnormal_loader = DataLoader(
    abnormal_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=1)

print("Strat training ...")
for epoch in range(opt.start_epoch, opt.end_epoch):
    print('epoch -- > [%d/%d] \t %s ' % (epoch, opt.end_epoch, time.ctime()))
    loss_total = 0
    j = 0
    for i, (batch) in enumerate(abnormal_loader, 0):
        F, T1, T1c, T2, uni_B = batch['F'].type(Tensor), batch['T1'].type(
            Tensor), batch['T1c'].type(Tensor), batch['T2'].type(Tensor), batch['uni_B'].type(Tensor)
        M, M_WT, M_ET, M_NET = batch['M'].type(Tensor), batch['M_WT'].type(
            Tensor), batch['M_ET'].type(Tensor), batch['M_NET'].type(Tensor)
        if torch.sum(M) < 100:
            continue
        level_Circle_M, uni_Circle_M = batch['level_Circle_M'].type(
            Tensor), batch['uni_Circle_M'].type(Tensor)

        # Tumor shape network update
        if opt.step == 0 or opt.step == 2:
            optm_tm_shape.zero_grad()
            out_mask_shape = tumor_shape(torch.cat([uni_B, Circle_M], 1))
            tumor_shape_loss = L1Loss(out_mask_shape, M_WT)
            tumor_shape_loss.backward()
            optm_tm_shape.step()
            # Tumor grade network update
            optm_tm_grade.zero_grad()
            out_mask_grade = tumor_grade(
                torch.cat([M_WT, level_Circle_M], 1))
            tumor_grade_loss = L1Loss(out_mask_grade, M)
            tumor_grade_loss.backward()
            optm_tm_grade.step()

        # Input and Output for Inpaint Network
        if opt.step == 1 or opt.step == 2:

            brain = torch.cat((F, T1, T1c, T2), 1)
            brain_blank = brain * (1 - M_WT)
            # print(brain_blank.shape, M.shape, torch.unique((M)), torch.unique(M_WT),torch.unique(M_ET),torch.unique(M_NET))
            out_brain = inp_gen(brain_blank, M)

            # optm_inp_dis.zero_grad()
            inp_dis_loss = realTargetLoss(
                inp_dis(brain, M)) + fakeTargetLoss(inp_dis(out_brain.detach(), M))
            inp_dis_loss.backward()
            optm_inp_dis.step()

            optm_inp_gen.zero_grad()
            inp_gb_loss = L1Loss(out_brain, brain)/torch.sum(uni_B)
            inp_lc_loss = L1Loss(out_brain*M_WT, brain*M_WT)/torch.sum(M_WT)
            inp_adv_loss = realTargetLoss(inp_dis(out_brain, M))

            t_F, t_T1, t_T1c, t_T2 = torch.split(
                brain*M_WT, split_size_or_sections=1, dim=1)
            ot_F, ot_T1, ot_T1c, ot_T2 = torch.split(
                out_brain*M_WT, split_size_or_sections=1, dim=1)

            t_F_ft, ot_F_ft = vgg(one2three(t_F).cuda()), vgg(
                one2three(ot_F).cuda())
            t_T1_ft, ot_T1_ft = vgg(one2three(t_T1).cuda()), vgg(
                one2three(ot_T1).cuda())
            t_T1c_ft, ot_T1c_ft = vgg(one2three(t_T1c).cuda()), vgg(
                one2three(ot_T1c).cuda())
            t_T2_ft, ot_T2_ft = vgg(one2three(t_T2).cuda()), vgg(
                one2three(ot_T2).cuda())
            content_loss = (L1Loss(t_F_ft.relu2_2, ot_F_ft.relu2_2)+L1Loss(t_T1_ft.relu2_2, ot_T1_ft.relu2_2)+L1Loss(
                t_T1c_ft.relu2_2, ot_T1c_ft.relu2_2)+L1Loss(t_T2_ft.relu2_2, ot_T2_ft.relu2_2))/torch.sum(uni_B)
            inp_gen_loss = inp_gb_loss + inp_lc_loss + content_loss + inp_adv_loss

            inp_gen_loss.backward()
            optm_inp_gen.step()

            loss_total += inp_gen_loss.data

    if (epoch) % 10 == 0:
        print("Saving the model ...")
        model_dir = os.path.join(opt.result, 'model/%d' % (epoch))
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
        torch.save(tumor_shape.state_dict(), model_dir + '/tumor_shape.pth')
        torch.save(tumor_grade.state_dict(), model_dir + '/tumor_grade.pth')
        torch.save(inp_gen.state_dict(), model_dir + '/inp_gen.pth')
