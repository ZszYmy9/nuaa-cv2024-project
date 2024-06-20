import os
from pathlib import Path

import numpy as np
from PIL import Image

from torch.utils import data
from torchvision import transforms

content_data = 'data/train2014'
style_data = 'data/train2014'

# ToTensor
def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


content_tf = train_transform()
style_tf = train_transform()

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print("training data from ", self.root)
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root, self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root, file_name)):
                    self.paths.append(self.root+"/"+file_name+"/"+file_name1)
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img
    def __len__(self):
        return len(self.paths)
    def name(self):
        return 'FlatFolderDataset'


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

# you can train our model using different dataset
content_dataset = FlatFolderDataset(content_data, content_tf)

style_dataset = FlatFolderDataset(style_data, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=8,
    sampler=InfiniteSamplerWrapper(content_dataset)
))

style_iter = iter(data.DataLoader(
    style_dataset, batch_size=8,
    sampler=InfiniteSamplerWrapper(style_dataset)
))