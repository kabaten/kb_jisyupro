import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import torch

from complement import complement
from kinematics import MyLink
from pyserial_servos2 import send

arm = MyLink(75, 75)

xs = torch.randint(60, 125, [5, 1]).to(torch.float)
ys = torch.randint(-35, 35, [5, 1]).to(torch.float)
points = torch.cat((xs, ys), dim=1)
print(points)
points = points.detach().numpy().T
print(points)
px, py = points
p = complement(points,100,2)
x, y = p

th = arm.ik(p)


th = th*180./np.pi
th = th.T
print(th.shape)

start = np.concatenate([th[0],[0]], 0)[None, :]
end = np.concatenate([th[-1],[0]], 0)[None, :]

H, W = th.shape
th_3 = np.full((H, 1), 25)
th = np.concatenate([th, th_3], 1)
print(th.shape)

th = np.concatenate([start, th, end], 0)
print(th.shape)
th = th.round() #.astype(np.int32)

# th = np.array([[45,  5,  25],# サーボ1, サーボ2は0度までは動けない.
#                     [90,  5,  25],
#                     [45,  5,  25],
#                     [ 5,  5,  25],
#                     [ 5, 45,  25],
#                     [ 5, 90,  25],
#                     [ 5, 45,  25],
#                     [ 5,  5,  25],
#                     [ 5,  5,  0]])

send(th)

plt.scatter(px, py)
plt.plot(x, y)
plt.show()