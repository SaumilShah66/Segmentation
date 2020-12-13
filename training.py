import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import random
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
from skimage import io, transform
import _pickle as pickle
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.models as models

class SegmentationData(data.Dataset):
    def __init__(self, file_name="Data.csv"):
        file = open(file_name, 'r') 
        self.data_ = file.readlines()
        random.shuffle(self.data_)
        print("Data length ",len(self.data_))
        file.close()
        pass
            
    def __len__(self): 
        return len(self.data_)
    
    def __getitem__(self, index):
        data = (self.data_[index]).split(',')
        image_name = data[0]
        x, y = int(data[1]), int(data[2])
        h, w = int(data[3]), int(data[4])
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        patch = image[x:h+x, y:w+y, :]/255.0
        id_class = int(data[5]) + 1
        id_ = np.zeros(10, dtype=np.float)
        id_[id_class] = 1
        sample = {'image': patch, 'id': id_}
        return sample

class Rescale(object):
    def __init__(self, output_size=32):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, id_ = sample['image'], sample['id']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        if new_h>self.output_size:
            img = img[:self.output_size,:,:]
        if new_w>self.output_size:
            img = img[:,:self.output_size,:]
        return {'image': img, 'id': id_}

class ToTensor(object):
    def __call__(self, sample):
        image, id_ = sample['image'], sample['id']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'id': torch.from_numpy(id_)}

class RandomFlip(object):
    def __call__(self, sample):
        if random.randint(0,1)==1:
            sample['image'] = cv2.flip(sample['image'], 1)
        return sample

class Normaliza(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        img = sample['image']
        img -= mean
        img /= std
        sample['image'] = img
        return sample

class Trans(object):
    def __init__(self, mean, std):
        self.norm = Normaliza(mean, std)
        self.flip = RandomFlip()
        self.tensor = ToTensor()
    
    def __call__(self, sample):
        return self.tensor(self.norm(self.flip(sample)))

# Model definition
class SegmentationNN(nn.Module):
    def __init__(self):
        resnet18 = models.resnet18(pretrained=True)
        pass
        
    def forward(self):
        pass

if __name__=="__main__":
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    data_transform = Trans(np.array(mean), np.array(std))           
    
    scale = Rescale(32)
    dataloader = SegmentationData(file_name="Data.csv")
    
    fig = plt.figure()
    for i in range(4):
        sample = scale(dataloader[i])
        new = {'image':sample['image'].copy(), 'id':sample['id'].copy()}
        final = data_transform(new)
        
        print(i, sample['image'].shape, np.argmax(sample['id']))
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Id #{}'.format(np.argmax(sample['id'])))
        ax.axis('off')
        ax.imshow(sample['image'])
        if i == 3:
            plt.show()
            break
        
        
