import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import albumentations as A
import cv2
class an_BrainDataset(Dataset):
    def __init__(self, root, mode, data_dir=None,aug = False):
        if mode == 'Test':
            self.F = sorted(glob.glob(os.path.join(root, mode, 'F') + '/*.png'))
            self.T1 = sorted(glob.glob(os.path.join(root, mode, 'T1') + '/*.png'))
            self.T1c = sorted(glob.glob(os.path.join(root, mode, 'T1c') + '/*.png'))
            self.T2 = sorted(glob.glob(os.path.join(root, mode, 'T2') + '/*.png'))
            self.M = sorted(glob.glob(os.path.join(root, mode, 'M') + '/*.png'))
            self.mode = mode


        else:
            self.F = sorted(glob.glob(os.path.join(root, mode, data_dir, 'F') + '/*.png'))
            self.T1 = sorted(glob.glob(os.path.join(root, mode, data_dir, 'T1') + '/*.png'))
            self.T1c = sorted(glob.glob(os.path.join(root, mode, data_dir, 'T1c') + '/*.png'))
            self.T2 = sorted(glob.glob(os.path.join(root, mode, data_dir, 'T2') + '/*.png'))
            self.M = sorted(glob.glob(os.path.join(root, mode, data_dir, 'M') + '/*.png'))
            self.mode = mode

        self.transform = None
        if aug == 'all':
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                ],
                additional_targets={'F': 'image', 'T1': 'image', 'T1c': 'image', 'T2': 'image', 'M_WT': 'mask',
                                    'M_ET': 'mask', 'M_NET': 'mask', 'M': 'mask'}
            )
        elif aug == 'rotate':
            self.transform = A.Compose(
                [
                    A.ShiftScaleRotate(p=0.5)
                ],
                additional_targets={'F': 'image', 'T1': 'image', 'T1c': 'image', 'T2': 'image', 'M_WT': 'mask',
                                    'M_ET': 'mask', 'M_NET': 'mask', 'M': 'mask'}
            )
        elif aug == 'bright_contrast':
            self.transform = A.Compose(
                [
                    A.RandomBrightnessContrast(p=0.2),
                ],
                additional_targets={'F': 'image', 'T1': 'image', 'T1c': 'image', 'T2': 'image', 'M_WT': 'mask',
                                    'M_ET': 'mask', 'M_NET': 'mask', 'M': 'mask'}
            )
        elif aug == 'flip':
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5)
                ],
                additional_targets={'F': 'image', 'T1': 'image', 'T1c': 'image', 'T2': 'image', 'M_WT': 'mask',
                                    'M_ET': 'mask', 'M_NET': 'mask', 'M': 'mask'}
            )


        self.to_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()])

    def __getitem__(self, index):
        F = cv2.imread(self.F[index])
        T1 = cv2.imread(self.T1[index])
        T1c = cv2.imread(self.T1c[index])
        T2 =  cv2.imread(self.T2[index])
        M = cv2.cvtColor(cv2.imread(self.M[index]), cv2.COLOR_RGB2GRAY) / 255
        M = self.qauntize(M)
        M_WT, M_ET, M_NET = self.level(M, 'WT'), self.level(M, 'ET'), self.level(M, 'NET')
        if self.transform is not None:
            aug = self.transform(image=F, F=F, T1=T1, T1c=T1c, T2=T2, M_WT=M_WT, M_ET=M_ET, M_NET=M_NET)
            F, T1, T1c, T2 = aug['F'], aug['T1'], aug['T1c'], aug['T2']
            M_WT, M_ET, M_NET = aug['M_WT'], aug['M_ET'], aug['M_NET']

        M_ALL = np.expand_dims(M_WT * 0.5 + M_ET * 0.25 + M_NET * 0.5, axis=0)
        M_WT = np.expand_dims(M_WT, axis=0)
        M_ET = np.expand_dims(M_ET, axis=0)
        M_NET = np.expand_dims(M_NET, axis=0)

        return {'F': self.to_tensor(F), 'T1':self.to_tensor(T1), 'T1c': self.to_tensor(T1c), 'T2': self.to_tensor(T2),
                'uni_B': self.to_tensor(self.unify(F)), 'M_WT': M_WT, 'M_ET': M_ET, 'M_NET': M_NET, 'M':M_ALL}

    def __len__(self):
        return len(self.M)

    def unify(self, image):
        temp = np.zeros(image.shape)
        temp[np.where(image > 0)] =  1.
        return np.uint8(temp) * 255

    def level(self, mask, mode):
        temp = np.zeros(mask.shape)
        if mode == 'ED':
            temp[np.where(mask == 0.5)] = 1.
        elif mode == 'ET':
            temp[np.where(mask == 0.75)] = 1.
        elif mode == 'NET':
            temp[np.where(mask == 1)] = 1.
        elif mode == 'WT':
            temp[np.where(mask > 0)] = 1.

        return temp


    def qauntize(self,mask):
        temp = np.zeros(mask.shape)
        temp[np.where(mask > 0.7)]= 1 # ET
        temp[np.where(mask > 0.35) and np.where(mask < 0.7)] = 0.5 # ED
        temp[np.where(mask > 0.15) and np.where(mask < 0.35)] = 0.25 # NET
        temp[np.where(mask < 0.2)] = 0

        # TRICK!
        temp2 = np.zeros(mask.shape)
        temp2[np.where(temp == 0.25)] = 1 # NET
        temp2[np.where(temp == 0.5)] = 0.5 # ED
        temp2[np.where(temp == 1)] = 0.75 # ET

        return temp2

    def circle(self, mask):
        [c, h, w]= mask.shape
        #find radius
        nonzero_idx = np.where(mask > 0)
        temp = np.array(nonzero_idx)
        l = temp.shape[1]
        radius = int((l/3.14)**(.5))
        #find center
        xx, yy = np.mgrid[:h, :w]

        x_center=np.sum(nonzero_idx[1])/(l+0.001)
        y_center=np.sum(nonzero_idx[2])/(l+0.001)

        circle = (xx - x_center) ** 2 + (yy - y_center) ** 2
        circle = circle < radius ** 2
        circle = np.array(np.reshape(circle, [c, h, w]), dtype=np.uint8)

        return circle

    def level_circle(self, mask):
        [c, h, w]= mask.shape
        # find radius
        l1 = np.array(np.where(mask > 0.24)).shape[1]
        radius_1 = int((l1 / 3.14) ** (.5))
        # find radius
        l2 = np.array(np.where(mask > 0.49)).shape[1]
        radius_2 = int((l2 / 3.14) ** (.5))
        # find radius
        l3 = np.array(np.where(mask > 0.74)).shape[1]
        radius_3 = int((l3 / 3.14) ** (.5))
        # find radius
        l4 = np.array(np.where(mask > 0.99)).shape[1]
        radius_4 = int((l4 / 3.14) ** (.5))

       # print(radius_1, radius_2, radius_3, radius_4)
        #find center
        xx, yy = np.mgrid[:h, :w]
        nonzero_idx = np.where(mask > 0)
        x_center=np.sum(nonzero_idx[1])/(l1+0.001)
        y_center=np.sum(nonzero_idx[2])/(l1+0.001)

        circle = (xx - x_center) ** 2 + (yy - y_center) ** 2

        temp1 = circle < radius_1**2
        temp1 = temp1*0.25
        temp2 = circle<radius_2 ** 2
        temp2 = temp2*0.25
        temp3 = circle<radius_3 ** 2
        temp3 = temp3*0.25
        temp4 = circle<radius_4 ** 2
        temp4 = temp4*0.25

        temp = temp1 + temp2 + temp3 + temp4

        result = np.array(np.reshape(temp, [c, h, w]))
        return result





#
# class an_BrainDataset(Dataset):
#     def __init__(self, root, mode, transforms_=None,randomcircle=False):
#         if mode == 'Augment':
#             self.transform = transforms.Compose(transforms_)
#             # self.F = sorted(glob.glob(os.path.join(root, 'Train/Original_part/2', 'F', ) + '/*.png'))
#             self.F = sorted(glob.glob(os.path.join(root, 'F', ) + '/*.png'))
#             self.T1 = sorted(glob.glob(os.path.join(root,'T1') + '/*.png'))
#             self.T1c = sorted(glob.glob(os.path.join(root,'T1c') + '/*.png'))
#             self.T2 = sorted(glob.glob(os.path.join(root, 'T2') + '/*.png'))
#             self.M = sorted(glob.glob(os.path.join(root,'M') + '/*.png'))
#             # self.M = sorted(glob.glob(os.path.join('/home/sunho/PycharmProjects/ksh/venv/data/Brain_v5/Train/Original_part/5/M') + '/*.png'))
#             self.randomcircle = randomcircle
#
#         else:
#             self.transform = transforms.Compose(transforms_)
#             self.F = sorted(glob.glob(os.path.join(root, mode, 'F',) + '/*.png'))
#             self.T1 = sorted(glob.glob(os.path.join(root, mode, 'T1') + '/*.png'))
#             self.T1c = sorted(glob.glob(os.path.join(root, mode, 'T1c') + '/*.png'))
#             self.T2 = sorted(glob.glob(os.path.join(root, mode, 'T2') + '/*.png'))
#             self.M = sorted(glob.glob(os.path.join(root, mode, 'M') + '/*.png'))
#             self.mode = mode
#             self.randomcircle= randomcircle
#
#     def __getitem__(self, index):
#         # angle = random.randint(-5, 5)
#         # F = self.transform(TF.rotate(Image.open(self.F[index]), angle))
#         # T1 = self.transform(TF.rotate(Image.open(self.T1[index]), angle))
#         # T1c = self.transform(TF.rotate(Image.open(self.T1c[index]), angle))
#         # T2 = self.transform(TF.rotate(Image.open(self.T2[index]), angle))
#         # M = self.transform(TF.rotate(Image.open(self.M[index]), angle))
#         F = self.transform(Image.open(self.F[index]))
#         T1 = self.transform(Image.open(self.T1[index]))
#         T1c = self.transform(Image.open(self.T1c[index]))
#         T2 = self.transform(Image.open(self.T2[index]))
#         M = self.transform(Image.open(self.M[index]))
#         level_circle = self.level_circle(M)
#
#         if self.randomcircle == True:
#             rand_index = random.randint(0, 1000)
#             M = self.qauntize(self.transform(Image.open(self.M[rand_index])))
#             level_circle = self.random_level_circle(M)
#
#         # return {'F': F, 'T1': T1, 'T1c': T1c, 'T2': T2, 'M': M}
#         return {'F': F, 'T1': T1, 'T1c': T1c, 'T2': T2,  'uni_Brain': self.unify(F),'M': M,\
#                 'Binary_M': self.unify(M), 'Circle': self.circle(M),'level_Circle':level_circle}
#
#         # return {'M': M,'Binary_M': self.unify(M), 'level_Circle': level_circle}
#
#     def __len__(self):
#         return len(self.F)
#
#     def unify(self, image):
#         c, h, w = image.shape
#         temp = np.zeros((c, h, w))
#         nonzero_idx = np.where(image > 0)
#         temp[nonzero_idx] = 1
#         unify_img = np.reshape(temp, [c, h, w])
#         return unify_img
#
#     def qauntize(self,mask):
#         c, h, w = mask.shape
#         temp = np.zeros((c, h, w))
#         idx = np.where(mask > 0)
#         temp[idx] = 0
#         idx = np.where(mask > 0.15)
#         temp[idx] = 0.25
#         idx = np.where(mask > 0.376)
#         temp[idx] = 0.50
#         idx = np.where(mask > 0.626)
#         temp[idx] = 0.75
#         idx = np.where(mask > 0.876)
#         temp[idx] = 1
#
#         # # TRICK!
#         temp2 = np.zeros((c,h,w))
#         temp2[np.where(temp==0.25)] =1          # level 1->4 # NET
#         temp2[np.where(temp == 0.5)] = 0.5      # level 2->2 # ED
#         temp2[np.where(temp == 1)] = 0.75       # level 4->3 # ET
#         temp2[np.where(temp == 0.75)] = 0.25    # level 3->1
#         return temp
#
#     def level(self,mask, l):
#         c, h, w = mask.shape
#         temp = np.zeros((c, h, w))
#
#         temp[np.where(mask==l)] = 1
#
#         return temp
#
#     def circle(self, mask):
#         [c, h, w]= mask.shape
#         #find radius
#         nonzero_idx = np.where(mask > 0)
#         temp = np.array(nonzero_idx)
#         l = temp.shape[1]
#         radius = int((l/3.14)**(.5))
#         #find center
#         xx, yy = np.mgrid[:h, :w]
#
#         x_center=np.sum(nonzero_idx[1])/(l+0.001)
#         y_center=np.sum(nonzero_idx[2])/(l+0.001)
#
#         circle = (xx - x_center) ** 2 + (yy - y_center) ** 2
#         circle = circle < radius ** 2
#         circle = np.array(np.reshape(circle, [c, h, w]), dtype=np.uint8)
#         return circle
#
#     def random_circle(self, mask):
#         [c, h, w]= mask.shape
#         #find radius
#         l = random.randint(100,2000)
#         radius = int((l/3.14)**(.5))
#         #find center
#         xx, yy = np.mgrid[:h, :w]
#
#         x_center= random.randint(40,210)
#         y_center= random.randint(40,210)
#
#         circle = (xx - x_center) ** 2 + (yy - y_center) ** 2
#         circle = circle < radius ** 2
#         circle = np.array(np.reshape(circle, [c, h, w]), dtype=np.uint8)
#         return circle
#
#     def level_circle(self, mask):
#         [c, h, w]= mask.shape
#         # find radius
#         l1 = np.array(np.where(mask > 0.24)).shape[1]
#         radius_1 = int((l1 / 3.14) ** (.5))
#         # find radius
#         l2 = np.array(np.where(mask > 0.49)).shape[1]
#         radius_2 = int((l2 / 3.14) ** (.5))
#         # find radius
#         l3 = np.array(np.where(mask > 0.74)).shape[1]
#         radius_3 = int((l3 / 3.14) ** (.5))
#         # find radius
#         l4 = np.array(np.where(mask > 0.99)).shape[1]
#         radius_4 = int((l4 / 3.14) ** (.5))
#
#        # print(radius_1, radius_2, radius_3, radius_4)
#         #find center
#         xx, yy = np.mgrid[:h, :w]
#         nonzero_idx = np.where(mask > 0)
#         x_center=np.sum(nonzero_idx[1])/(l1+0.001)
#         y_center=np.sum(nonzero_idx[2])/(l1+0.001)
#
#         circle = (xx - x_center) ** 2 + (yy - y_center) ** 2
#
#         temp1 = circle < radius_1**2
#         temp1 = temp1*0.25
#         temp2 = circle<radius_2 ** 2
#         temp2 = temp2*0.25
#         temp3 = circle<radius_3 ** 2
#         temp3 = temp3*0.25
#         temp4 = circle<radius_4 ** 2
#         temp4 = temp4*0.25
#
#         temp = temp1 + temp2 + temp3 + temp4
#
#         result = np.array(np.reshape(temp, [c, h, w]))
#         return result
#
#     def random_level_circle(self, mask):
#         [c, h, w]= mask.shape
#         # find radius
#         l1 = np.array(np.where(mask > 0.24)).shape[1]
#         if l1 < 4:
#             l1, l2, l3, l4 = 0, 0, 0, 0
#         else:
#             ran_num = random.sample(range(0,l1), 3)
#             ran_num.sort()
#             [l4, l3, l2] = ran_num
#
#         radius_1 = int((l1 / 3.14) ** (.5))
#         radius_2 = int((l2 / 3.14) ** (.5))
#         radius_3 = int((l3 / 3.14) ** (.5))
#         radius_4 = int((l4 / 3.14) ** (.5))
#        # print(radius_1, radius_2, radius_3, radius_4)
#         #find center
#         xx, yy = np.mgrid[:h, :w]
#         nonzero_idx = np.where(mask > 0)
#         x_center=np.sum(nonzero_idx[1])/(l1+0.001)
#         y_center=np.sum(nonzero_idx[2])/(l1+0.001)
#
#         circle = (xx - x_center) ** 2 + (yy - y_center) ** 2
#
#         temp1 = circle < radius_1**2
#         temp1 = temp1*0.25
#         temp2 = circle<radius_2 ** 2
#         temp2 = temp2*0.25
#         temp3 = circle<radius_3 ** 2
#         temp3 = temp3*0.25
#         temp4 = circle<radius_4 ** 2
#         temp4 = temp4*0.25
#
#         temp = temp1 + temp2 + temp3 + temp4
#
#         result = np.array(np.reshape(temp, [c, h, w]))
#         return result
