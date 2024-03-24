import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from skimage import data,img_as_float
from skimage.util import random_noise

def cylinder_reconsitution(data):
    data=np.reshape(data,(449,199))
    data[data>5]=5
    data[data < -5] = -5

    return data[0:384,4:196]

def norml(data):
    data=(data+5)/10
    return data

class CylinderDataset(Dataset):
    def __init__(self,data,noise_data,if_norm=False):
        super(CylinderDataset, self).__init__()
        self.data=data
        self.noise_data=noise_data
        self.if_norm=if_norm
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        input=self.data[index]
        noise_input=self.noise_data[index]
        if self.if_norm:
            input=norml(input)
            noise_input=norml(noise_input)
        input = cylinder_reconsitution(input)
        noise_input=cylinder_reconsitution(noise_input)
        return np.array([input]),np.array([noise_input])


def norml_pattern(data):
    data=(data+2)/6
    return data

class pattern_Dataset(Dataset):
    def __init__(self,data,noisy_data,if_norm=False):
        super(pattern_Dataset, self).__init__()
        self.data=data
        self.noisy_data=noisy_data
        self.if_norm = if_norm
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        input=self.data[index].T
        noise_input=self.noisy_data[index].T
        if self.if_norm:
            input=norml_pattern(input)
            noise_input=norml_pattern(noise_input)
        return np.array([input]),np.array([noise_input])
