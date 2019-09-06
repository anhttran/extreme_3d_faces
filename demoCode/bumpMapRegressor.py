from __future__ import print_function, division
import torch
import glob
import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from skimage import io
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as utils
import torchvision.transforms as transforms
import torch.nn.init as init
import time
import copy
import cv2
import sys
import torch.nn.functional as F

class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        np1 = np.array([[[[-1, 1],[0, 0]]]],'f')
	self.op1 = Variable(torch.from_numpy(np1).cuda())
        np2 = np.array([[[[-1, 0],[1, 0]]]],'f')
	self.op2 = Variable(torch.from_numpy(np2).cuda())

    def forward(self, image, target):
        image1 = F.conv2d(image, self.op1, padding=0)
        target1 = F.conv2d(target, self.op1, padding=0)
        image2 = F.conv2d(image, self.op2, padding=0)
        target2 = F.conv2d(target, self.op2, padding=0)
        criterionL1 = nn.L1Loss()
        return criterionL1(image1,target1) + criterionL1(image2, target2)

class BumpsDataset(Dataset):
    def __init__(self,imList, transform=None, type='test', generate=False):
        #In case the text aren't found or generate text files again
        if type == 'test':
            self.name = 'test_set'
            fin = open(imList, 'r')
            lines = fin.readlines()
            fin.close()
        else:
            print ("illegal dataset name")
            exit()

        self.pairs = []
        for l in lines:
            self.pairs.append(l[:-1])
        self.transform=transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        c = self.pairs[idx]
        imName = c.split('/')[-1]
        imName = '.'.join(imName.split('.')[:-1])
        image=io.imread(c)
	image = cv2.resize(image, (500,500))
        if len(image.shape) < 3:
            image = np.stack((image,image,image),axis=-1)
        if image.shape[2] == 1:
            image = np.concatenate((image,image,image),axis=2)
        if self.transform:
            image = self.transform(image)
        sample = {'name': imName, 'image': image}
        return sample


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, stride=1, padding=1, activation = nn.ReLU(), downsample=True):
        super(UNetConvBlock, self).__init__()
        self.conv_down = nn.Conv2d(in_size, in_size, kernel_size, stride=2, padding=1)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size,stride=1, padding=1)
        init.xavier_normal(self.conv_down.weight,gain=np.sqrt(2))
        init.xavier_normal(self.conv.weight,gain=np.sqrt(2))
        init.xavier_normal(self.conv2.weight,gain=np.sqrt(2))
        init.constant(self.conv_down.bias,0.1)
        init.constant(self.conv.bias, 0.1)
        init.constant(self.conv2.bias, 0.1)

        self.activation = activation
        self.downsample = downsample

    def forward(self, x, bridge=None):
        out=x
        if self.downsample:
            out = self.activation(self.conv_down(out))
        if bridge is not None:
            out = torch.cat([out, bridge], 1)
        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,stride=1, padding=1, activation=nn.ReLU(), space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_size, out_size, 3, stride=1, padding=1)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size,stride=1, padding=1)
        init.xavier_normal(self.conv0.weight,gain=np.sqrt(2))
        init.xavier_normal(self.conv.weight,gain=np.sqrt(2))
        init.xavier_normal(self.conv2.weight,gain=np.sqrt(2))
        init.constant(self.conv0.bias, 0.1)
        init.constant(self.conv.bias, 0.1)
        init.constant(self.conv2.bias, 0.1)
        self.activation = activation
        self.upsampler = nn.Upsample(scale_factor=2)


    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        up = self.upsampler(x)     # Up-sample
        up = self.conv0(up)           # Conv
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)
        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))
        return out

class UNet(nn.Module):
    def __init__(self, _unet = None):
        super(UNet, self).__init__()
        if _unet:
                self.conv_block1_16 = copy.deepcopy(_unet.conv_block1_16)
                self.conv_block16_32 = copy.deepcopy(_unet.conv_block16_32)
                self.conv_block32_64 = copy.deepcopy(_unet.conv_block32_64)
                self.conv_block64_128 = copy.deepcopy(_unet.conv_block64_128)
                self.conv_block128_256 = copy.deepcopy(_unet.conv_block128_256)
                self.conv_block256_512 = copy.deepcopy(_unet.conv_block256_512)
                self.conv_block512_1024 = copy.deepcopy(_unet.conv_block512_1024)

                self.up_block1024_512 = copy.deepcopy(_unet.up_block1024_512)
                self.up_block512_256 = copy.deepcopy(_unet.up_block512_256)
                self.up_block256_128 = copy.deepcopy(_unet.up_block256_128)
                self.up_block128_64 = copy.deepcopy(_unet.up_block128_64)
                self.up_block64_32 = copy.deepcopy(_unet.up_block64_32)
                self.up_block32_16 = copy.deepcopy(_unet.up_block32_16)
                self.up_block32_16_mask = copy.deepcopy(_unet.up_block32_16_mask)
        
                self.last_alt = copy.deepcopy(_unet.last_alt)
                self.last_alt_mask = copy.deepcopy(_unet.last_alt_mask)
        else:
                self.conv_block1_16 = UNetConvBlock(3, 16,padding=7, downsample=False)
                self.conv_block16_32 = UNetConvBlock(16, 32)
                self.conv_block32_64 = UNetConvBlock(32, 64)
                self.conv_block64_128 = UNetConvBlock(64, 128)
                self.conv_block128_256 = UNetConvBlock(128, 256)
                self.conv_block256_512 = UNetConvBlock(256, 512)
                self.conv_block512_1024 = UNetConvBlock(512, 1024)

                self.up_block1024_512 = UNetUpBlock(1024, 512)
                self.up_block512_256 = UNetUpBlock(512, 256)
                self.up_block256_128 = UNetUpBlock(256, 128)
                self.up_block128_64 = UNetUpBlock(128, 64)
                self.up_block64_32 = UNetUpBlock(64, 32)
                self.up_block32_16 = UNetUpBlock(32, 16)
                self.up_block32_16_mask = UNetUpBlock(32, 16)
                
                self.last_alt = nn.Conv2d(16,1,3,1,1)
                self.last_alt_mask = nn.Conv2d(16,1,3,1,1)
        self.tanh = nn.Tanh()


    def forward(self, x):
        block1 = self.conv_block1_16(x)  # -> 16x512x512
        block2 = self.conv_block16_32(block1) # -> 32x256x256
        block3 = self.conv_block32_64(block2) # -> 64x128x128
        block4 = self.conv_block64_128(block3) # -> 128x64x64
        block5 = self.conv_block128_256(block4) # -> 256x32x32
        block6 = self.conv_block256_512(block5) # -> 512x16x16
        block7 = self.conv_block512_1024(block6) # -> 1024x8x8
        up1 = self.up_block1024_512(block7, block6) # -> 512x16x16
        up2 = self.up_block512_256(up1, block5)
        up3 = self.up_block256_128(up2, block4)
        up4 = self.up_block128_64(up3, block3)
        up5 = self.up_block64_32(up4, block2)
        up6 = self.up_block32_16(up5, block1)
        up6_mask = self.up_block32_16_mask(up5, block1)
        ##################################
        return self.tanh(self.last_alt_mask(up6_mask)), self.tanh(self.last_alt(up6))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=2,padding=1)   # 256x256
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)    # 128x128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)    # 64x64
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)    # 32x32
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)    # 16x16
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)    # 8x8
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)    # 4x4
        self.conv8 = nn.Conv2d(512, 1024, kernel_size=4, stride=1, padding=0)    # 1x1
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(512)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        init.xavier_normal(self.conv1.weight, gain=np.sqrt(2))
        init.constant(self.conv1.bias, 0.1)
        init.xavier_normal(self.conv2.weight, gain=np.sqrt(2))
        init.constant(self.conv2.bias, 0.1)
        init.xavier_normal(self.conv3.weight, gain=np.sqrt(2))
        init.constant(self.conv3.bias, 0.1)
        init.xavier_normal(self.conv4.weight, gain=np.sqrt(2))
        init.constant(self.conv4.bias, 0.1)
        init.xavier_normal(self.conv5.weight, gain=np.sqrt(2))
        init.constant(self.conv5.bias, 0.1)
        init.xavier_normal(self.conv6.weight, gain=np.sqrt(2))
        init.constant(self.conv6.bias, 0.1)
        init.xavier_normal(self.conv7.weight, gain=np.sqrt(2))
        init.constant(self.conv7.bias, 0.1)
        init.xavier_normal(self.conv8.weight, gain=np.sqrt(2))
        init.constant(self.conv8.bias, 0.1)

    def forward(self,x):
        out = self.lrelu(self.bn1(self.conv1(x)))
        out = self.lrelu(self.bn2(self.conv2(out)))
        out = self.lrelu(self.bn3(self.conv3(out)))
        out = self.lrelu(self.bn4(self.conv4(out)))
        out = self.lrelu(self.bn5(self.conv5(out)))
        out = self.lrelu(self.bn6(self.conv6(out)))
        out = self.lrelu(self.bn7(self.conv7(out)))
        out = self.sigmoid(self.conv8(out))
        return out

class Net(nn.Module):
    def __init__(self, impath, outpath):
        super(Net, self).__init__()
        self.impath = impath
        self.outpath = outpath

        self.params={
            'Glr':2e-3,
            'Dlr':2e-3,
            'lr_steps':[15,40,60,80,100,120,140],
            'batch_size':1
        }

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_bumps = None #BumpsDataset('/media/anh/EXTRA/3D_CNN/SfS_train', transform=self.transform, type='train')
        self.train_dataloader = None #DataLoader(train_bumps, batch_size=self.params['batch_size'], shuffle=True, num_workers=4)
        test_bumps = BumpsDataset(impath, transform=self.transform, type='test')
        self.test_dataloader = DataLoader(test_bumps, batch_size=1, shuffle=False, num_workers=1)

        self.Dnet = Discriminator()
        self.Gnet = UNet()
        self.Goptimizer = optim.SGD(self.Gnet.parameters(), lr=self.params['Glr'])

    def forward(self,x,m,l):
        self.recon_mask, self.recon = self.Gnet.forward(x) #bump map reconstruction
        self.d_fake = self.Dnet.forward(self.recon.detach()) #probability of classifying reconstruction as fake
        self.d_real = self.Dnet.forward(l) #probability of classifying bump map as real

    def backwardG(self,m,l):
        criterionL1 = nn.L1Loss()
        criterionBCE = nn.BCELoss()
        self.g_fake = self.Dnet.forward(self.recon)
        self.GBCE_loss = criterionBCE(self.g_fake, Variable(torch.ones(self.g_fake.size()).cuda()))
                
                #Reconstruction loss
        self.GL1_loss = criterionL1(self.recon,l) + self.GLoss(self.recon,l)
        self.GL1_mask_loss = criterionL1(self.recon_mask,m) + self.GLoss(self.recon_mask,m)

        self.G_loss = 0.1*self.GL1_mask_loss + self.GL1_loss #+ 0.01*self.GBCE_loss
        self.G_loss.backward()

    def backwardD(self):
        criterionBCE = nn.BCELoss()
                #classification loss
        self.D_real_loss = criterionBCE(self.d_real, Variable(torch.ones(self.d_real.size()).cuda()))
        self.D_fake_loss = criterionBCE(self.d_fake, Variable(torch.zeros(self.d_real.size()).cuda()))
        self.D_loss = self.D_real_loss + self.D_fake_loss
        self.D_loss.backward()


    def optimize(self,x,m,l):
        self.forward(x,m,l)

        self.Goptimizer.zero_grad()
        self.backwardG(m,l)
        self.Goptimizer.step()

        ##self.Doptimizer.zero_grad()
        ##self.backwardD()
        ##self.Doptimizer.step()

    def reconstruct_test(self,epoch):
        for i,batch in enumerate(self.test_dataloader):
            images = batch['image']
            images = images.float()
            bumps = batch['bump']
            bumps = bumps.float()
            masks = batch['mask']
            masks = masks.float()
            images = Variable(images.cuda())
            recon_mask, recon = self.Gnet.forward(images)
            output = torch.cat((masks,recon_mask.data.cpu(),bumps,recon.data.cpu()),dim=3)
            utils.save_image(output, net.outpath + '/'+str(epoch)+'.'+str(i)+'.jpg',nrow=4, normalize=True)


    def save_images(self):
        for i,batch in enumerate(self.test_dataloader):
            images = batch['image']
            images = images.float()
	    imName = batch['name'][0]
            images = Variable(images.cuda(), volatile=True)
            recon_mask, recon = self.Gnet.forward(images)
	    reconmat = recon.data.cpu().numpy()
	    reconmat = (reconmat[0,0,:,:]+1)*255/2
	    reconmat_mask = recon_mask.data.cpu().numpy()
	    reconmat_mask = reconmat_mask[0,0,:,:]*255
	    cv2.imwrite(net.outpath + '/'+imName + '_bump.png',reconmat)
	    cv2.imwrite(net.outpath + '/'+imName + '_mask.png',reconmat_mask)

def estimateBump(modelPath, imList, outPath):
    global net
    net = Net(imList, outPath)
    net.cuda()
    net.load_state_dict(torch.load(modelPath));
    net.save_images()
