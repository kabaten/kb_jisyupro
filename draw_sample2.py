import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader

from generator import *
from discriminator import PlotLine2, Discriminator3
from trainer import train1
from hiragana_dataset import make_hiragana_dataset
from complement import complement
from kinematics import MyLink
from pyserial_servos2 import send

DATADIR = "../data/hiragana73"
CATEGORIES = ['a']
train_dataset, valid_dataset = make_hiragana_dataset(DATADIR, CATEGORIES)
# バッチサイズの指定
batch_size = 1
# DataLoaderを作成
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 動作確認
# イテレータに変換
batch_iterator = iter(train_dataloader)
# 1番目の要素を取り出す
inputs, labels = next(batch_iterator)
# sns.heatmap(labels.detach().numpy()[0])
# plt.show()


img_size = 48
n_point = 20

a_gen = Generator6(img_size, n_point, w_m=0.25, w_M=9.)
# disc = Discriminator3(img_size)
# losses, errors, distances, distributions = train1(gen, disc, inputs, labels, iteration=2000, lr=0.005)

a_gen.load_state_dict(torch.load('gen_a_params.pth', map_location=torch.device('cpu')))
a_gen = a_gen.eval()
xyw = a_gen(inputs)

plotline = PlotLine2(48)
line = plotline(xyw)
# sns.heatmap(line.detach().numpy()[0])
# plt.show()

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
print(xy.shape)
xy = xy[[1,0]]
#breakpoint()

W = 75/48*np.array([[1],[-1]])
b = np.array([[50],[37.5]])

xy = W*xy + b

px, py = xy
p = complement(xy,100,1)
x, y = p
arm = MyLink(75, 75)
th12 = arm.ik(p)


th12 = th12*180./np.pi
th12 = th12.T
print(th12.shape)
#breakpoint()

start = np.concatenate([th12[0],[0]], 0)[None, :]
end = np.concatenate([th12[-1],[0]], 0)[None, :]

H, W = th12.shape
#th_3 = np.full((H, 1), 30)
th = np.concatenate([th12, th3[:, None]], 1)
print(th.shape)

th = np.concatenate([start, th, end], 0)
print(th.shape)
th = th.round()

send(th)

# plt.scatter(px, py)
# plt.plot(x, y)
# plt.show()
sns.heatmap(line.detach().numpy()[0])
plt.show()