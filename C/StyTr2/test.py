import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from thop import profile
from torch import nn
from torch.utils import data
from torchvision import transforms
from torchvision.utils import save_image
from collections import OrderedDict
# from tqdm import tqdm

import StyTR
import transformer

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

vgg = StyTR.vgg
vgg.load_state_dict(torch.load("vgg_normalised.pth"))
vgg = nn.Sequential(*list(vgg.children())[:44])


decoder = StyTR.decoder
embedding = StyTR.PatchEmbed()
Trans = transformer.Transformer()
decoder.eval()
embedding.eval()
Trans.eval()

parser = argparse.ArgumentParser()
args = parser.parse_args()
network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)
input1, input2 = torch.rand(1, 3, 512, 512), torch.rand(1, 3, 512, 512)
print(profile(network, inputs=(input1, input2, ))[0]/1e9)

state_dict = torch.load('decoder_iter_160000.pth')
new_state_dict = OrderedDict()
for k,v in state_dict.items():
    namekey=k
    new_state_dict[namekey]=v
decoder.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load('transformer_iter_160000.pth')
for k,v in state_dict.items():
    namekey=k
    new_state_dict[namekey]=v
Trans.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load('embedding_iter_160000.pth')
for k,v in state_dict.items():
    namekey=k
    new_state_dict[namekey]=v
embedding.load_state_dict(new_state_dict)

parser = argparse.ArgumentParser()
args = parser.parse_args()
network = StyTR.StyTrans(vgg, decoder, embedding, Trans,args)



network.eval()


network.to(device)

content_tf = train_transform()
style_tf = train_transform()

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root,self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root,file_name)):
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

content_dataset = FlatFolderDataset('content', content_tf)
style_dataset = FlatFolderDataset('style', style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=1,
    sampler=InfiniteSamplerWrapper(content_dataset)
))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=1,
    sampler=InfiniteSamplerWrapper(style_dataset)
))

c_a = 0
s_a = 0

for i in range(20):
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    with torch.no_grad():
        out, loss_c, loss_s, l_identity1, l_identity2 = network(content_images, style_images)
    output_name = '{:s}/{:s}{:s}'.format(
            "res", str(i), ".jpg"
    )
    print(i)
    c_a = c_a + loss_c
    s_a = s_a + loss_s

    out = torch.cat((content_images, out), 0)
    out = torch.cat((style_images, out), 0)
    out = out.to(torch.device('cpu'))
    save_image(out, output_name)


