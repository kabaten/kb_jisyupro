import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

from generator import *
from discriminator import PlotLine2, Discriminator3
from trainer import train1
from hiragana_dataset import make_hiragana_dataset, make_input3
from complement import complement
from kinematics import MyLink
from pyserial_servos2 import send

tgt = 'u'

mask = nn.Transformer.generate_square_subsequent_mask(16).T
gen_ = Generator8_2(w_m=0, w_M=4, mask=mask)
gen_.load_state_dict(torch.load("../model/" + tgt + "/best.pt"))

inputs = torch.load("../model/" + tgt + "/input.pt")

xyw = gen_(inputs)
# xyw = torch.tensor([[[23.5, 23.5, 1.],
#                      [  47,   47, 1.1],
#                      [  47,    0, 1.2],
#                      [   0,    0, 1.1],
#                      [   0,   47, 1.]]])

plotline = PlotLine2(48)
line = plotline(xyw)
#sns.heatmap(line.detach().numpy()[0])
#plt.show()
#breakpoint()

xy = xyw[0, :, :2]
w = xyw[0, :, 2]

s = torch.cat((torch.zeros(1,), torch.sqrt(((torch.diff(xy, dim=-2))**2).sum(dim=-1))))
sw = torch.stack((s,w))
sw = sw.detach().numpy()
_, w = complement(sw,100,1)
w = 1.8*w + 2.5
th3 = 5.6*w + 18



# xs = torch.randint(60, 125, [5, 1]).to(torch.float) # x方向には35mm分板が飛び出ている. 補完のことを考えると, 50~60mmの余裕が必要か.
# ys = torch.randint(-35, 35, [5, 1]).to(torch.float)
# points = torch.cat((xs, ys), dim=1)
# print(points)

xy = xy.detach().numpy().T
#print(xy.shape)
xy = xy[[1,0]]
#breakpoint()

xy = xy + np.array([[-23.5],[-23.5]])

#W = 75/47*np.array([[1.0],[-1.2]])
W = 75/47*np.array([[1.0],[-1.0]])
#b = np.array([[50.],[37.5]])
b = np.array([[87.5],[0]])

xy = W*xy + b
#print(xy)
#breakpoint()
#xy = xy + b
#xy = xy + np.array([[12.5],[0]])

# xy = xy + np.array([[10],[5]])
# W = 75/48*np.array([[1],[-1]])
# b = np.array([[50],[37.5]])

# xy = W*xy + b

px, py = xy
p = complement(xy,100,1)
x, y = p
arm = MyLink(75, 75)
th12 = arm.ik(p)


th12 = th12*180./np.pi
th12 = th12.T + np.array([[-11,-4]]) #-8,-5
#print(th12)
#breakpoint()

start = np.concatenate([th12[0],[0]], 0)[None, :]
end = np.concatenate([th12[-1],[0]], 0)[None, :]

H, W = th12.shape
#th_3 = np.full((H, 1), 30)
th = np.concatenate([th12, th3[:, None]], 1)
#print(th.shape)

th = np.concatenate([start, th, end], 0)
#print(th.shape)
th = th.round()

send(th)

# plt.scatter(px, py)
# plt.plot(x, y)
# plt.show()
sns.heatmap(line.detach().numpy()[0])
plt.show()