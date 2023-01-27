import torch
import torch.nn as nn
import numpy as np
import math

class PositionalEncoding(nn.Module):
    """
    モデルに入力するデータの時系列的順番に関する情報を埋め込むためのクラス
    """
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Batch First の場合は以下
        # token_embedding = torch.transpose(token_embedding, 0, 1)
        # return torch.transpose(self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :]), 0, 1)
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class Generator(nn.Module):
    """
    点の数までチャンネル数を増やし, transformerへ img_size=48, n_point=16(固定)
    w_m: 太さの最小値(の指標)
    w_M: 太さの最大値(の指標)
    mask: 過去の情報にマスクすると良い
    """
    def __init__(self, w_m=0., w_M=4., mask=None):
        super().__init__()
        self.img_size = 48
        self.n_point = 16
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, padding=2),nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, padding=2),nn.ReLU(),nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, padding=2),nn.ReLU(),nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2),nn.ReLU(),nn.MaxPool2d(2, stride=2)
        )

        self.positional_encoder = PositionalEncoding(dim_model=36, max_len=20, dropout_p=0.1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=36, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=3, norm=nn.LayerNorm(36))

        self.linear_mapping = nn.Linear(in_features=36, out_features=3)

        self.sigmoid = nn.Sigmoid()

        self.flatten = nn.Flatten()

        self.p_scale = float(self.img_size - 1)
        self.w_M = w_M
        self.w_m = w_m
        self.w_scale = w_M - w_m
        self.scale_W = nn.Parameter(torch.tensor([[[self.p_scale, self.p_scale, self.w_scale]]]), requires_grad=False)
        self.scale_b = nn.Parameter(torch.tensor([[[0., 0., w_m]]]), requires_grad=False)

        if mask is not None:
            self.mask = nn.Parameter(mask, requires_grad=False)
        else:
            self.mask = None
        
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], 16, 36)
        x = x.permute(1,0,2)
        x = self.positional_encoder(x)

        x = self.encoder(x, mask=self.mask)
        y = self.sigmoid(self.linear_mapping(x))

        raw = y.permute(1,0,2)
        out = raw*self.scale_W + self.scale_b
        return out

class PlotLine(nn.Module):
    """
    点を指定された太さの直線で順に結んだ画像を生成するクラス
    input: [N, points_num, 3]
    """
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

class Discriminator(nn.Module):
    """
    評価器
    """
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

        self.plot_line = PlotLine(self.img_size)
        self.criterion = nn.BCELoss()
        
        self.error_log = []
        self.distance_log = []
        
    def forward(self, points, img):
        N = points.shape[0]
        x = ((torch.diff(points[:,:,0:2], dim=-2))**2).sum(dim=-1)
        sqrt_x = self.c0 + self.c1*(x - self.x0) + self.c2*(x - self.x0)**2 + self.c3*(x - self.x0)**3
        distance = sqrt_x.sum()/N
        
        line = self.plot_line(points)
        error = self.criterion(line, img)*self.img_size**2
        loss = error + distance
        
        self.error_log.append(error.detach().cpu().numpy())
        self.distance_log.append(distance.detach().cpu().numpy())
        
        return loss