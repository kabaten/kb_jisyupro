import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sys
sys.path.append('..')

from layers import *
from trainer import *
from data import make_input

## train関数の仕様上ひとつ上のディレクトリにフォルダを作る必要がある。
# os.mkdir("../model")
# os.mkdir("../model/house")

path = 'house.jpg'
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
src = (~image)/255
inputs = torch.from_numpy(make_input(src)).type(torch.float32)
labels = inputs[:, -1]

torch.save(inputs, "../model/house/input.pt")

img_size = 48
plotline = PlotLine(img_size)
disc = Discriminator(img_size)

mask = nn.Transformer.generate_square_subsequent_mask(16).T

gen = Generator(w_m=0, w_M=4, mask=mask) #bridge a600b50
losses, errors, distances = train(
    gen, disc, inputs, labels, 
    iteration=5000, start_factor=0.1, end_factor=0.8, total_iters=40, 
    start_p=0.5, alpha=400, beta=50,
    folder="house"
)

gen_ = Generator(w_m=0, w_M=4, mask=mask)
gen_.load_state_dict(torch.load("../model/house/best.pt"))

xyw = gen_(inputs)
line = plotline(xyw)
sns.heatmap(line.detach().numpy()[0])
plt.show()