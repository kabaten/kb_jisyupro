"""
ロボットを動かして描画する
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import sys
sys.path.append('..')

from soft.layers import *
from soft.trainer import *
from soft.data import *
from complement import complement
from kinematics import MyLink
from serial2servos import send

tgt = 'u'

mask = nn.Transformer.generate_square_subsequent_mask(16).T
gen_ = Generator(w_m=0, w_M=4, mask=mask)
gen_.load_state_dict(torch.load("../model/" + tgt + "/best.pt"))

inputs = torch.load("../model/" + tgt + "/input.pt")

xyw = gen_(inputs)

plotline = PlotLine(48)
line = plotline(xyw)

xy = xyw[0, :, :2]
w = xyw[0, :, 2]

s = torch.cat((torch.zeros(1,), torch.sqrt(((torch.diff(xy, dim=-2))**2).sum(dim=-1))))
sw = torch.stack((s,w))
sw = sw.detach().numpy()
_, w = complement(sw,100,1)
w = 1.8*w + 2.5
th3 = 5.6*w + 18

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

sns.heatmap(line.detach().numpy()[0])
plt.show()