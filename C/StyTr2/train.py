import argparse

import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
from sample import content_iter, style_iter

import StyTR
import transformer


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=7.0)
parser.add_argument('--save_model_interval', type=int, default=10000)
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

vgg = StyTR.vgg
vgg.load_state_dict(torch.load("vgg_normalised.pth"))
vgg = nn.Sequential(*list(vgg.children())[:44])


decoder = StyTR.decoder
embedding = StyTR.PatchEmbed()

Trans = transformer.Transformer()
with torch.no_grad():
    network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)
network.train()

network.to(device)

content_tf = train_transform()
style_tf = train_transform()

optimizer = torch.optim.Adam([
    {'params': network.transformer.parameters()},
    {'params': network.decode.parameters()},
    {'params': network.embedding.parameters()},
], lr=args.lr)


loss_sum = []

for i in range(16000):

    if i < 1e4:
        warmup_learning_rate(optimizer, iteration_count=i)
    else:
        adjust_learning_rate(optimizer, iteration_count=i)

    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)

    out, loss_c, loss_s, l_identity1, l_identity2 = network(content_images, style_images)
    if (i+1) % 10 == 0:
        print("train_epoch:{:d}".format(i+1))
    if (i+1) % 1000 == 0:
        output_name = '{:s}{:s}'.format(
             str(i), ".jpg"
        )

        out = torch.cat((content_images, out), 0)
        out = torch.cat((style_images, out), 0)
        out = out.to(torch.device('cpu'))
        save_image(out, output_name)

    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s + (l_identity1 * 70) + (l_identity2 * 1)
    if (i+1)% 1000 ==0:
        loss_sum.append(loss)
        print("train_epoch:{:d},loss:{:f}".format(i+1,loss))
    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()
    if (i+1) == 5000 or (i+1) == 6000:
        state_dict = network.transformer.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   'transformer_iter_{:d}.pth'.format(i + 1))

        state_dict = network.decode.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   'decoder_iter_{:d}.pth'.format(i + 1))

        state_dict = network.embedding.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   'embedding_iter_{:d}.pth'.format(i + 1))


