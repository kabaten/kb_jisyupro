import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import torch

import sys, os
sys.path.append('..')

from generator import *
from discriminator import PlotLine2

tgt = input("dir:")

mask = nn.Transformer.generate_square_subsequent_mask(16).T
gen_ = Generator8_2(w_m=0, w_M=4, mask=mask)
gen_.load_state_dict(torch.load("../model/" + tgt + "/best.pt"))

inputs = torch.load("../model/" + tgt + "/input.pt")

xyw = gen_(inputs)

plotline = PlotLine2(img_size=48)

fig = plt.figure()

def init():
    sns.heatmap(np.zeros([48,48]), vmax=1., cbar=False)

def animate(i):
    line_i = plotline(xyw[:, :i+2, :])
    sns.heatmap(line_i.detach().numpy()[0], vmax=1., cbar=False)

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=15, repeat=True, interval=500)

anim.save("../" + tgt + ".gif")