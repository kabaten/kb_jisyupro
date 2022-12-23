import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.append('..')
import cv2

import seaborn as sns

class Conv2Central(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel  = Variable(torch.tensor([[[[1., 0.5],[0.5, 0.25]]]]))
    
    def forward(self, img):
        N, H, W = img.size()
        img0 = img.unsqueeze(1)
        
        img1 = torch.cat((torch.cat((img0, torch.zeros(N, 1, 1, W)), dim=2), torch.zeros(N, 1, H+1, 1), ), dim=3)
        img2 = F.conv2d(img1, self.kernel, stride=(1, 1))
        img3 = torch.flip(img2, dims=[0])
        
        img4 = torch.cat((torch.cat((img3, torch.zeros(N, 1, 1, W)), dim=2), torch.zeros(N, 1, H+1, 1), ), dim=3)
        img5 = F.conv2d(img4, self.kernel, stride=(1, 1))
        img6 = torch.flip(img5, dims=[1])
        
        img7 = torch.cat((torch.cat((img6, torch.zeros(N, 1, 1, W)), dim=2), torch.zeros(N, 1, H+1, 1), ), dim=3)
        img8 = F.conv2d(img7, self.kernel, stride=(1, 1))
        img9 = torch.flip(img8, dims=[0])
        
        img10 = torch.cat((torch.cat((img9, torch.zeros(N, 1, 1, W)), dim=2), torch.zeros(N, 1, H+1, 1), ), dim=3)
        img11 = F.conv2d(img10, self.kernel, stride=(1, 1))
        img12 = torch.flip(img11, dims=[1])
        
        img13 = img12.squeeze(1)
        return img13

class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, g):
        return g

class PlotLine2(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        #self.conv2central = Conv2Central()
#         self.num2point = Num2Point(img_size)
#         self.num2point.load_state_dict(torch.load('params.pth', map_location=torch.device('cpu') ))
#         # freeze all layers
#         for param in self.num2point.parameters():
#             param.requires_grad = False
        self.cs = torch.tensor(range(0, img_size)).to(torch.float).unsqueeze(0)
    
    def forward(self, points):
        N = points.size()[0] # batch_size
        
        points0 = torch.cat((torch.zeros([N,1,2]),
                            points.repeat([1,1,2]).reshape([N,-1, 2]),
                            torch.zeros([N,1,2])), dim=1).reshape([N, -1, 2, 2])[:, 1:-1]
        #print(points0)
        t = torch.tensor(range(0, self.img_size+1))/self.img_size
        ts = torch.stack((1-t, t)).to(torch.float)
        line = torch.matmul(ts.mT, points0)
        #print(line)
        #line = torch.round(line)
        line0 = RoundNoGradient.apply(line)
        #line0 = line
        #print(line0)
        line1 = line0[:, :, 0:-1].reshape(N, -1, 2)
        #print(line1)
        shape = line1.size()
        line2 = line1.flatten()
        #line4 = F.softmax(self.num2point(line2), dim=1)
        line4 = torch.exp(-(line2.reshape(N, -1, 1) - self.cs)**2)
        #print(line4)
        line5 = line4.reshape(*shape, -1)
        x = line5[:, :, 0, :].unsqueeze(2)
        y = line5[:, :, 1, :].unsqueeze(2)
        p_map = torch.matmul(x.mT, y).sum(dim=1)
        #p_map0 = self.conv2central(p_map)
        p_map0 = torch.tanh(p_map)
        #p_map0 = p_map
        return p_map0

def process_img2(img):
    img_np = np.array(img)
    img_np = ~img_np
    _, img_th = cv2.threshold(img_np, 0, 1, cv2.THRESH_OTSU)
#     kernel = np.ones((5,5),np.float32)/25.
#     img_blr = cv2.filter2D(img_th.astype(np.float32),-1,kernel)
    # img_blr = img_blr.squeeze()
    return img_th

class Discriminator2(nn.Module):
    def __init__(self, img):
        super().__init__()
        self.img_size = len(img)
        self.img = torch.from_numpy(process_img2(img)).clone().unsqueeze(0)
        
        self.plot_line = PlotLine2(self.img_size)
        #self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.p_map = None
        
    def forward(self, points):
        alpha = 0.06 #0.06
        beta = 1.75 #1.00
        distance = (torch.diff(points, dim=-2)**2).sum()
        step_distances = (torch.diff(points, dim=-2)**2).sum(dim=-1)
        step_mean = step_distances.mean()
#         var = points.var(dim=-2).sum()
        var = points.std(dim=-2).sum()
#         var = torch.log(points.var(dim=-2).sum())
#         var = 1.
        eps = 1e-5
        const = distance/(var + eps)
        
        x = self.plot_line(points)
        self.p_map = x
        t = self.img
        #loss = torch.sum(x*self.img)
        #loss = self.criterion(x, self.img)
        #loss = ((x - t)**2).sum() + distance
        #loss = ((x - t)**2).sum() + const
        loss = (beta*((x - t)**2).sum() + alpha*distance)/(var + eps)
#         loss = (beta*((x - t)**2).sum())/(var + eps) + alpha*distance
        return loss

class Generator2(nn.Module):
    def __init__(self, img_size, n_point):
        super().__init__()
        self.n_point = n_point
        self.img_size = img_size
        
        self.flatten = nn.Flatten()
        
        input_size = img_size**2
        output_size = 2*n_point
        hidden_size1 = img_size*output_size
        hidden_size2 = n_point**2
        
        self.fc1 = nn.Linear(input_size, hidden_size1, bias=False)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2, bias=False)
        self.fc3 = nn.Linear(hidden_size2, output_size, bias=False)
        #self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, img):
        x = self.flatten(img)
        #x = x.view(-1, 1) # Linear層に入力できるようにサイズを(batch_size, C)にする. --> x = nn.Flatten(img)
        x = self.sigmoid(self.fc1(x)) # write me! # fc1 + sigmoid
        x = self.sigmoid(self.fc2(x)) # write me! # fc2 + sigmoid
        y = self.fc3(x) # write me! # fc3
        y_num = torch.sigmoid(y)*float(self.img_size - 1.0)
        out = y_num.reshape(-1, self.n_point, 2)
        return out

def zscore(x, axis = None):
    x_mean = x.mean(axis=axis, keepdims=True)
    x_std  = np.std(x, axis=axis, keepdims=True)
    z_score = (x-x_mean)/x_std
    return z_score

def make_input(img):
    img = np.array(img)
    img = ~img
    img = zscore(img)
    return torch.from_numpy(img).to(torch.float)

def train_gen2(img, n_point=20, iteration = 2000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using: {device}')
    
    img_size = len(img)
    
    gen = Generator2(img_size, n_point).to(device)
    disc = Discriminator2(img).to(device)
    
    optimizer=optim.SGD(gen.parameters(), lr=0.001)
    
    points_log = []
    losses = []

    inputs = make_input(img).unsqueeze(0)
    inputs = inputs.to(device)

    gen = gen.train()

    for i in range(0, iteration):
        points = gen(inputs)
        
        loss = disc(points)
        
        optimizer.zero_grad()
    
        loss.backward()
    
        optimizer.step()
    
        if i%100 == 0:
            print(f'iter: {i}, loss: {loss}')
            points_log.append(points.cpu())
            losses.append(loss.cpu())
            
    return points_log, losses