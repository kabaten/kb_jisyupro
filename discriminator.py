import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class PlotLine2(nn.Module):# 太さも操作可能
    # input: [N, points_num, 3]
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size

        self.cmp_num = int(img_size*np.sqrt(2))
        self.co = nn.Parameter(torch.tensor(range(0, img_size)).to(torch.float).unsqueeze(0), requires_grad=False) # 座標ベクトル
        t = torch.tensor(range(0, self.cmp_num+1))/self.cmp_num
        self.ts = nn.Parameter(torch.stack((1-t, t)).to(torch.float), requires_grad=False)# 補完ベクトル
        self.zeros = nn.Parameter(torch.zeros([1,1,3]), requires_grad=False)
    
    def forward(self, points):
        N = points.size()[0] # batch_size
        
        point_pairs = torch.cat((self.zeros.repeat([N,1,1]),
                                 points.repeat([1,1,2]).reshape([N,-1,3]),
                                 self.zeros.repeat([N,1,1])),dim=1).reshape([N, -1, 2, 3])[:, 1:-1]
        
        segs_points = torch.matmul(self.ts.mT, point_pairs)
        
        line_points = segs_points[:, :, 0:-1].reshape(N, -1, 3)
        
        # print(shape)
        xy = line_points[:,:,:2]
        xy_shape = xy.size()
        # print(xy_shape)
        w = line_points[:,:,2].unsqueeze(-1).unsqueeze(-1)
        # print(w.size())
        
        xy_ver = xy.reshape(N, -1, 1)

        plot_xy = torch.exp((-(xy_ver - self.co)**2).reshape(*xy_shape, -1)/(2*w))
        # print(plot_xy.size())

        x = plot_xy[:, :, 0, :].unsqueeze(2)
        y = plot_xy[:, :, 1, :].unsqueeze(2)

        draft = torch.matmul(x.mT, y).sum(dim=1)
        line = torch.tanh(draft)
        return line

class Discriminator3(nn.Module):# 距離計算にテイラー展開を利用 *
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        
        d = np.array(range(-img_size+1, img_size))
        x0 = (d**2 + (d**2).T).mean().astype(np.float32)
        self.x0 = float(x0)
        self.c0 = float(x0**(0.5))
        self.c1 = float(x0**(-0.5)/2.)
        self.c2 = float(-(x0**(-1.5)/8.))
        self.c3 = float(x0**(-2.5)/16.)

        self.plot_line = PlotLine2(self.img_size)
        self.criterion = nn.BCELoss()
        
        self.error_log = []
        self.distance_log = []
        
    def forward(self, points, img):
        N = points.shape[0]
        x = ((torch.diff(points[:,:,0:2], dim=-2))**2).sum(dim=-1)
        # x = ((torch.diff(points, dim=-2))**2).sum(dim=-1)
        sqrt_x = self.c0 + self.c1*(x - self.x0) + self.c2*(x - self.x0)**2 + self.c3*(x - self.x0)**3
        distance = sqrt_x.sum()/N
        
        line = self.plot_line(points)
        error = self.criterion(line, img)*self.img_size**2
        loss = error + distance
        
        self.error_log.append(error.detach().cpu().numpy())
        self.distance_log.append(distance.detach().cpu().numpy())
        
        return loss

class Discriminator4(nn.Module):# 距離計算にテイラー展開を利用 part2
    def __init__(self, img_size, n_point):
        super().__init__()
        self.img_size = img_size
        self.n_point = n_point
        
        d = np.array(range(-img_size+1, img_size))
        x0 = (d**2 + (d**2).T).mean().astype(np.float32)
        self.x0 = float(x0)
        self.c0 = float(x0**(0.5))
        self.c1 = float(x0**(-0.5)/2.)
        self.c2 = float(-(x0**(-1.5)/8.))
        self.c3 = float(x0**(-2.5)/16.)

        self.plot_line = PlotLine2(self.img_size)
        self.criterion = nn.BCELoss()
        
        self.error_log = []
        self.distance_log = []
        self.distribution_log = []
        
    def forward(self, points, img):
        N = points.shape[0]
        x = ((torch.diff(points[:,:,0:2], dim=-2))**2).sum(dim=-1)
        # x = ((torch.diff(points, dim=-2))**2).sum(dim=-1)
        sqrt_x = self.c0 + self.c1*(x - self.x0) + self.c2*(x - self.x0)**2 + self.c3*(x - self.x0)**3
        distance = sqrt_x.sum()/N/self.n_point

        var = points[:,:,0:2].std(dim=-2).sum(dim=-1)
        
        line = self.plot_line(points)
        error = self.criterion(line, img)*self.img_size**2
        loss = error + distance
        
        self.error_log.append(error.detach().numpy())
        self.distance_log.append(distance.detach().numpy())
        self.distribution_log.append(var.detach().numpy())
        
        return loss

class Discriminator5(nn.Module):# 二画目にいかない問題 充填率で割った→性能が下がっただけ
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        
        d = np.array(range(-img_size+1, img_size))
        x0 = (d**2 + (d**2).T).mean().astype(np.float32)
        self.x0 = float(x0)
        self.c0 = float(x0**(0.5))
        self.c1 = float(x0**(-0.5)/2.)
        self.c2 = float(-(x0**(-1.5)/8.))
        self.c3 = float(x0**(-2.5)/16.)

        self.plot_line = PlotLine2(self.img_size)
        self.criterion = nn.BCELoss()
        
        self.error_log = []
        self.distance_log = []
        
    def forward(self, points, img):
        N = points.shape[0]
        x = ((torch.diff(points[:,:,0:2], dim=-2))**2).sum(dim=-1)
        # x = ((torch.diff(points, dim=-2))**2).sum(dim=-1)
        sqrt_x = self.c0 + self.c1*(x - self.x0) + self.c2*(x - self.x0)**2 + self.c3*(x - self.x0)**3
        distance = sqrt_x.sum()/N
        
        line = self.plot_line(points)
        error = self.criterion(line, img)*self.img_size**2

        hit = (img*line).sum()/img.sum()

        loss = (error + distance)/hit
        
        self.error_log.append(error.detach().cpu().numpy())
        self.distance_log.append(distance.detach().cpu().numpy())
        
        return loss

class Discriminator6(nn.Module):# 二画目にいかない問題 元画像で1なのに埋まってないところにペナルティを課す
    def __init__(self, img_size, alpha=0.25):
        super().__init__()
        self.img_size = img_size
        
        d = np.array(range(-img_size+1, img_size))
        x0 = (d**2 + (d**2).T).mean().astype(np.float32)
        self.x0 = float(x0)
        self.c0 = float(x0**(0.5))
        self.c1 = float(x0**(-0.5)/2.)
        self.c2 = float(-(x0**(-1.5)/8.))
        self.c3 = float(x0**(-2.5)/16.)

        self.plot_line = PlotLine2(self.img_size)
        self.criterion = nn.BCELoss()

        self.relu = nn.ReLU()
        self.alpha = alpha
        
        self.error_log = []
        self.distance_log = []
        
    def forward(self, points, img):
        N = points.shape[0]

        x = ((torch.diff(points[:,:,0:2], dim=-2))**2).sum(dim=-1)
        sqrt_x = self.c0 + self.c1*(x - self.x0) + self.c2*(x - self.x0)**2 + self.c3*(x - self.x0)**3
        distance = sqrt_x.sum()/N
        
        line = self.plot_line(points)
        error = self.criterion(line, img)*self.img_size**2

        miss = ((self.relu(img - line))*(-torch.log(line + 1e-7))).sum()/N

        loss = distance + (error + self.alpha * miss)/(1 + self.alpha)
        
        self.error_log.append(error.detach().cpu().numpy())
        self.distance_log.append(distance.detach().cpu().numpy())
        
        return loss

class Discriminator7(nn.Module):# 二画目にいかない問題 hitで距離を割る --だめ　--> trainerで対策
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        
        d = np.array(range(-img_size+1, img_size))
        x0 = (d**2 + (d**2).T).mean().astype(np.float32)
        self.x0 = float(x0)
        self.c0 = float(x0**(0.5))
        self.c1 = float(x0**(-0.5)/2.)
        self.c2 = float(-(x0**(-1.5)/8.))
        self.c3 = float(x0**(-2.5)/16.)

        self.plot_line = PlotLine2(self.img_size)
        self.criterion = nn.BCELoss()

        self.relu = nn.ReLU()
        
        self.error_log = []
        self.distance_log = []
        
    def forward(self, points, img):
        N = points.shape[0]

        x = ((torch.diff(points[:,:,0:2], dim=-2))**2).sum(dim=-1)
        sqrt_x = self.c0 + self.c1*(x - self.x0) + self.c2*(x - self.x0)**2 + self.c3*(x - self.x0)**3
        distance = sqrt_x.sum()/N
        
        line = self.plot_line(points)
        error = self.criterion(line, img)*self.img_size**2

        tgt = img.sum()
        #miss = ((self.relu(img - line))*(-torch.log(line + 1e-7))).sum()
        miss = (self.relu(img - line)).sum()
        hit = tgt - miss
        hit_rate = hit/tgt
        
        loss = distance*hit_rate + error/hit_rate
        
        self.error_log.append(error.detach().cpu().numpy())
        self.distance_log.append(distance.detach().cpu().numpy())
        
        return loss