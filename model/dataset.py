import os
import numpy as np
import torch
import cv2
from skimage import io
from glob import glob

# noinspection PyUnresolvedReferences
class Dataset(torch.utils.data.Dataset):
    def __init__(self, datadir, sample=-1, device='cpu', down=False,patch_size=None,nums_projects=None):
        self.sample = sample
        self.device = device

        imgfiles = sorted(glob(os.path.join(datadir, 'sino', '*.tif')))
        tarfiles = sorted(glob(os.path.join(datadir, 'slice', '*.tif')))
        index = np.arange(len(imgfiles))
        if self.sample != -1: 
            index = index[::round(len(imgfiles)/self.sample)]
        else: 
            index = np.setdiff1d(index, index[::round(len(imgfiles)/2)])
        self.index = index
        self.imgfiles = sorted(np.array(imgfiles)[index])
        self.tarfiles = sorted(np.array(tarfiles)[index])
        self.patch_size=patch_size
        self.idx=np.arange(1200)[::round(1200/nums_projects)]
    def __getitem__(self, index):
        imgfile, tarfile = self.imgfiles[index], self.tarfiles[index]
        img, tar = io.imread(imgfile), io.imread(tarfile)
        img=img[self.idx]
        tar[tar<-0.003]=-0.003
        tar[tar>0.003]=0.003

        img = (img-img.min())/(img.max()-img.min())    
        tar = (tar-tar.min())/(tar.max()-tar.min())
        if self.patch_size:
            input_patches, target_patches = get_patch(img,tar,self.patch_size)
            img=input_patches
            tar=target_patches
        img = torch.tensor(img.astype(np.float32)).unsqueeze(0)
        tar = torch.tensor(tar.astype(np.float32)).unsqueeze(0)
        if self.sample != -1: return img.to(self.device), tar.to(self.device)
        else: return img.to(self.device), tar.to(self.device),imgfile

    def __len__(self):
        return len(self.imgfiles)
def get_patch(full_input_img,full_target_img,patch_size):
        assert full_input_img.shape == full_target_img.shape
        patch_input_imgs = []
        patch_target_imgs = []
        h, w = full_input_img.shape
        new_h, new_w = patch_size, patch_size
        patch_n=int(h/new_h)
        for i in range(patch_n):
            top = i*new_h
            left = i*new_w
            patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
            patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
            patch_input_imgs.append(patch_input_img)
            patch_target_imgs.append(patch_target_img)
        return np.array(patch_input_imgs), np.array(patch_target_imgs)