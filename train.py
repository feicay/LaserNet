import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import visdom
from torch.utils import data
from torch.autograd import Variable
from data.datagen import Lidar_xyzic_dataset
from model.DLA import DLA
from model.loss import FocalLossClassify
import time
import argparse
import numpy as np
import os
import gc
import cv2

parser = argparse.ArgumentParser(description='PyTorch LaserNet Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--vis', default=1, type=int, help='visdom')
parser.add_argument('--test', default=0, type=int, help='test')
parser.add_argument('--onnx', default=0, type=int, help='onnx')
args = parser.parse_args()

max_epoch = 70

def train():
    start_epoch = 0
    if args.vis:
        vis = visdom.Visdom(env=u'test1')

    trainlist = '/raid/pytorch_ws/LaserNet/trainlist.txt'
    validlist = '/raid/pytorch_ws/LaserNet/validlist.txt'
    trainset = Lidar_xyzic_dataset(trainlist)
    validset = Lidar_xyzic_dataset(validlist, train=0)
    loader_train = data.DataLoader(trainset, batch_size=16, shuffle=1, num_workers=4, drop_last=True)
    loader_val = data.DataLoader(validset, batch_size=4, shuffle=1, num_workers=4, drop_last=True)

    network = DLA(8)
    if args.resume:
        print('Resuming from checkpoint..')
        checkpoint = torch.load('./checkpoint/lasernet45.pth')
        network.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch'] + 1
    net = torch.nn.DataParallel(network).cuda()
    criterion = FocalLossClassify(8)
    lr = args.lr
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    for i in range(start_epoch, max_epoch):
        print('--------start training epoch %d --------'%i)
        loss_train = 0.0
        net.train()
        for ii, (image, cls_truth) in enumerate(loader_train):
            #input
            image = Variable(image).cuda()
            cls_truth = Variable(cls_truth).cuda()
            #forward
            optimizer.zero_grad()
            t0 = time.time()
            cls_pred = net(image)
            t1 = time.time()
            #loss
            loss = criterion(cls_pred, cls_truth)
            t2 = time.time()
            #backward
            loss.backward()
            t3 = time.time()
            #update
            optimizer.step()
            t4 = time.time()
            loss_train += loss.data
            print('forward time: %f, loss time: %f, backward time: %f, update time: %f'%((t1-t0),(t2-t1),(t3-t2),(t4-t3)))
            print('%3d/%3d => loss: %f'%(ii,i,criterion.loss))
            if args.vis:
                vis.line(Y=loss.data.cpu().view(1,1).numpy(),X=np.array([ii]),win='loss',update='append' if ii>0 else None)
        if i < 3:
            loss_train = loss.data
        else:
            loss_train = loss_train / ii
        loss_val = 0.0
        net.eval()
        for jj, (image, cls_truth) in enumerate(loader_val):
            image = Variable(image).cuda()
            cls_truth = Variable(cls_truth).cuda()
            optimizer.zero_grad()
            cls_pred = net(image)
            loss = criterion(cls_pred, cls_truth)
            loss_val += loss.data
            print('val: %3d/%3d => loss: %f'%(jj,i,criterion.loss))
        loss_val = loss_val / jj
        if args.vis:
            vis.line(Y=torch.cat((loss_val.view(1,1), loss_train.view(1,1)),1).cpu().numpy(),X=np.array([i]),\
                        win='eval-train loss',update='append' if i>0 else None)
        print('Saving weights..')
        state = {
            'net': net.module.state_dict(),
            'loss': loss_val,
            'epoch': i,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/lasernet%d.pth'%i)
        del image, cls_truth
        del cls_pred
        gc.collect()
        time.sleep(1)
        if i==50:
            lr = lr*0.1
            print('learning rate: %f'%lr)
            for para_group in optimizer.param_groups:
                para_group['lr'] = lr
    torch.save(network,'lasernet_model_final.pkl')
    print('finish training!')

def test():
    color = np.array([[0, 0, 0],
                  [0, 0, 250],
                  [0, 250, 250],
                  [0, 250, 0],
                  [250, 250, 0],
                  [250, 0, 0],
                  [250, 0, 250],
                  [150, 150, 150]])
    validlist = '/raid/pytorch_ws/LaserNet/validlist.txt'
    validset = Lidar_xyzic_dataset(validlist, train=0)
    loader_test = data.DataLoader(validset, batch_size=1, shuffle=1, num_workers=1, drop_last=True)

    network = DLA(8)
    checkpoint = torch.load('./checkpoint/lasernet69.pth')
    network.load_state_dict(checkpoint['net'])
    network = network.cuda().eval()
    if args.onnx == 1:
        dummy_input = torch.randn(4, 3, 200, 400, device='cuda')
        torch.onnx.export(network, dummy_input, "lasernet.onnx", verbose=True)
        return
    for i, (image, cls_truth) in enumerate(loader_test):
        image = Variable(image).cuda()
        cls_truth = Variable(cls_truth).cuda()
        t1 = time.time()
        cls_pred = network(image)
        t2 = time.time()
        print('inference time %f'%(t2-t1))
        print(cls_pred.size())
        im = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        cv2.imshow('image', im)
        pred = F.softmax(cls_pred, dim=1).squeeze(0).cpu()
        prob, cls_ = pred.max(dim=0)
        im_cls = np.zeros((200, 400, 3), dtype=np.uint8)
        im_cls[:,:] = color[cls_[:,:]]
        im_truth = np.zeros((200, 400, 3), dtype=np.uint8)
        im_truth[:,:] = color[cls_truth[:,:].cpu().numpy()]
        cv2.imshow('cls', im_cls)
        cv2.imshow('truth', im_truth)
        cv2.waitKey(0)

if args.test:
    test()
else:
    train()