import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import torch

from complement import complement
from kinematics import MyLink
from pyserial_servos2 import send

arm = MyLink(75, 75)

xs = torch.randint(60, 125, [5, 1]).to(torch.float) # x方向には35mm分板が飛び出ている. 補完のことを考えると, 50~60mmの余裕が必要か.
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
th = th.round()

send(th)

plt.scatter(px, py)
plt.plot(x, y)
plt.show()