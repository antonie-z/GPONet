import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
import os
import cv2
import numpy as np
from torch import nn
import matplotlib.pyplot as plt


class Normalize(object):
    def __init__(self):
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])

    def __call__(self, image, mask=None, edge=None):
        image = (image - self.mean) / self.std
        if mask is None:
            return image
        if edge is None:
            return image, mask / 255
        return image, mask / 255, edge / 255

class RandomCrop(object):
    def __call__(self, image, mask=None, edge=None):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        if mask is None:
            return image[p0:p1, p2:p3, :]
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], edge[p0:p1, p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask=None, edge=None):
        if np.random.randint(2) == 0:
            if mask is None:
                return image[:, ::-1, :].copy()
            return image[:, ::-1, :].copy(), mask[:, ::-1].copy(), edge[:, ::-1].copy()
        else:
            if mask is None:
                return image
            return image, mask, edge

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None, edge=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if edge is None:
            return image, mask
        edge = cv2.resize(edge, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, edge

class ToTensor(object):
    def __call__(self, image, mask=None, edge=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image

        mask = torch.from_numpy(mask)
        if edge is None:
            return image, mask

        edge = torch.from_numpy(edge)
        return image, mask, edge


class TrainDataset(Dataset):
    def __init__(self, root:str, train:bool=True, aug = True):
        assert os.path.exists(root), f"path '{root}' does not exist."
        self.train = train
        self.aug = aug
        if train:
            self.image_root = os.path.join(root, 'DUTS-TR', 'DUTS-TR-Image')
            self.mask_root = os.path.join(root, 'DUTS-TR', 'DUTS-TR-Mask')
            self.edge_root = os.path.join(root, 'DUTS-TR', 'DUTS-TR-Detail')
        else:
            self.image_root = os.path.join(root, 'DUTS-TE', 'DUTS-TE-Image')
            self.mask_root = os.path.join(root, 'DUTS-TE', 'DUTS-TE-Mask')
            self.edge_root = os.path.join(root, 'DUTS-TE', 'DUTS-TE-Detail')

        image_names = [p for p in os.listdir(self.image_root) if p.endswith(".jpg")]
        mask_names = [p for p in os.listdir(self.mask_root) if p.endswith(".png")]
        edge_names = [p for p in os.listdir(self.edge_root) if p.endswith(".png")]

        # check images and masks
        re_mask_names = []
        for p in image_names:
            mask_name = p.replace('.jpg','.png')
            assert mask_name in mask_names, f"{p} has no corresponding mask"
            re_mask_names.append(mask_name)
        mask_names = re_mask_names

        self.image_path = [os.path.join(self.image_root,n) for n in image_names]
        self.mask_path = [os.path.join(self.mask_root,n) for n in mask_names]
        self.edge_path = [os.path.join(self.edge_root,n) for n in edge_names]

        self.normalize = Normalize()
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(352, 352)
        self.totensor = ToTensor()
        self.img_names = image_names


    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        mask_path = self.mask_path[idx]
        edge_path = self.edge_path[idx]
        img_name = self.img_names[idx]

        image = cv2.imread(image_path,flags=cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB

        h, w, _ = image.shape

        mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        edge = cv2.imread(edge_path, flags=cv2.IMREAD_GRAYSCALE)

        image, mask, edge = self.normalize(image, mask, edge)
        if self.aug:
            image, mask, edge = self.randomcrop(image, mask, edge)
            image, mask, edge = self.randomflip(image, mask, edge)
        image, mask, edge = self.resize(image, mask, edge)
        image, mask, edge = self.totensor(image, mask, edge)
        mask = torch.unsqueeze(mask,dim=0)
        edge = torch.unsqueeze(edge,dim=0)


        if self.train:
            return image, mask, edge
        else:
            return image, mask, edge, str(img_name)

    def __len__(self):
        return len(self.image_path)



