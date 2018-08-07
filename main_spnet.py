from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
from math import log10

import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import torchvision
import time
import os, os.path
# from logger import Logger
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model_spnet import CNN_spnet, clip_grad_value_
from data import get_training_set, get_test_set, count_parameters

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', default = True, help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--tag', type=int, default=1, help='1 is for high frequency, 2 is for low frequency')
parser.add_argument('--dataset', type=str, default='./Train/Patch/', help='')
parser.add_argument('--checkpoints', type=int, default=0, help='load trained model, 0 is not')
parser.add_argument('--num_features', type=int, default=10, help='')
parser.add_argument('--filter_size', type=int, default=5, help='')
parser.add_argument('--inp_chans', type=int, default=3, help='')
parser.add_argument('--nlayers', type=int, default=1, help='')
parser.add_argument('--seq_len', type=int, default=4, help='')
parser.add_argument('--name', type=str, default='main_spnet', help='')
opt = parser.parse_args()

print(opt)

def to_np(x):
    return x.cpu().data.numpy()

cuda = opt.cuda
print(cuda)


if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor,opt.dataset)
test_set = get_test_set(opt.upscale_factor,opt.dataset)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

num_features=opt.num_features
filter_size=opt.filter_size
batch_size=opt.batchSize
shape=(84/opt.upscale_factor,84/opt.upscale_factor)#H,W
inp_chans=opt.inp_chans
nlayers=opt.nlayers
seq_len=opt.seq_len

print('===> Building model')
if opt.checkpoints == 0:
    model = CNN_spnet(upscale_factor=opt.upscale_factor)
else:
    model = torch.load('./checkpoints/'+ opt.name + 'model_epoch_{}.pth'.format(opt.checkpoints))

criterion = nn.MSELoss()
l1criterion = nn.MSELoss()
clip_value = 1e-2
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()
    l1criterion =l1criterion.cuda()

print('Parameters:')
param = count_parameters(model)
print(param)

optimizer = optim.Adam(model.parameters(), lr=opt.lr)

def train(epoch):
    epoch_loss = 0
    input_global, target_global, pred_global = 0, 0, 0
    tag = 3
    for iteration, batch in enumerate(training_data_loader, 1):
        # hard samples training
        input, target = Variable(batch[0]), Variable(batch[1])
        input_bilinear = torch.nn.functional.upsample(input, scale_factor = opt.upscale_factor, mode='bilinear')
        target_new = torch.cat((target,target),2)
        shape=(input.size(2),input.size(3))

        if cuda:
            input = input.cuda()
            target = target.cuda()
            target_new = target_new.cuda()
        optimizer.zero_grad()
        pred = model(input, tag)
        
        loss = l1criterion(pred, target_new)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / (len(training_data_loader)) ))
    step = epoch * (opt.nEpochs + 1)


def test():
    avg_psnr = 0
    avg_psnr_base = 0
    avg_mse = 0
    im_num = 1
    t_all = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda()
            target = target.cuda()
        
        input_bilinear = torch.nn.functional.upsample(input, scale_factor = opt.upscale_factor, mode='bilinear')
            
        tag = 3
        t1 = time.clock()
        prediction = model(input, tag)
        t2 = time.clock()
        t_all = t_all + (t2 - t1)
        
        prediction = prediction.narrow(1,1,1)
        
        mse = criterion(prediction, target)
        mse_bil = criterion(input_bilinear, target)
        if mse.data[0] == 0 or mse_bil.data[0] == 0:
            continue

        psnr = 10 * log10(1 / mse.data[0])
        mse_all = mse.data[0] - mse_bil.data[0]
        psnr_bil = 10 * log10(1 / mse_bil.data[0])
        avg_psnr_base += psnr_bil
        avg_mse += mse_all
        avg_psnr += psnr

    print("The testing time is %f second" % (t_all))
    print("===> Avg. PSNR: {:.4f} dB   Bilinear {:.4f} dB ".format(avg_psnr / (len(testing_data_loader)), avg_psnr_base / (len(testing_data_loader))))

def checkpoint(epoch):
    model_out_path = './checkpoints/' + opt.name + 'model_epoch_{}.pth'.format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if opt.checkpoints == 0:
    start_epoch = 1
else:
    start_epoch = opt.checkpoints

for epoch in range(start_epoch, opt.nEpochs + 1):
    train(epoch)
    if epoch % 100 == 0 or epoch == 2:
        test()
    if epoch % 100 == 0:
        checkpoint(epoch)
