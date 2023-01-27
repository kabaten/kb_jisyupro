import torch
import torch.nn as nn
import math
#import torch.optim as optim
#import torch.nn.functional as F
#from torch.autograd import Variable

class Generator2(nn.Module):# 太さ対応生成器(1)
    def __init__(self, img_size, n_point, w_m=0.25, w_M=4.):
        super().__init__()
        self.n_point = n_point
        self.img_size = img_size
        
        self.flatten = nn.Flatten()
        
        input_size = img_size**2
        output_size = 3*n_point
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, stride=2)

        conved = (img_size - 4)//2
        conved = (conved - 4)//2
        hidden_size1 = 3*conved**2
        print(hidden_size1)
        
        hidden_size2 = 3*((conved**2 + output_size)//2)
        print(hidden_size2)

        self.fc1 = nn.Linear(hidden_size1, hidden_size2, bias=False)
        self.fc2 = nn.Linear(hidden_size2, output_size, bias=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.p_scale = float(img_size - 1)
        self.w_M = w_M
        self.w_m = w_m
        self.w_scale = w_M - w_m
        self.scale_W = torch.tensor([[[self.p_scale, self.p_scale, self.w_scale]]])
        self.scale_b = torch.tensor([[[0., 0., w_m]]])
        
    def forward(self, data):
        x = self.relu(self.conv1(data)) # conv1 -> relu
        x = self.pool(x) # pool
        x = self.relu(self.conv2(x)) # conv2 -> relu
        x = self.pool(x) # pool
        x = x.view(x.size(0), -1) # reshape to use linear function
        x = self.sigmoid(self.fc1(x)) # fc1 -> sigmoid
        y = self.sigmoid(self.fc2(x)) # fc1 -> sigmoid
        raw = y.reshape(-1, self.n_point, 3)
        out = raw*self.scale_W + self.scale_b
        return out

class Generator3(nn.Module):# 太さ対応生成器(2)*
    def __init__(self, img_size, n_point, w_m=0.25, w_M=4.):
        super().__init__()
        self.n_point = n_point
        self.img_size = img_size
        
        self.flatten = nn.Flatten()
        
        input_size = img_size**2
        output_size = 3*n_point
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, stride=2)

        conved = (img_size - 4)//2
        conved = (conved - 4)//2
        hidden_size1 = 3*conved**2
        print(hidden_size1)
        
        hidden_size2 = 3*((conved**2 + output_size)//2)
        #hidden_size2 = 3*((conved**2 + n_point)//2)
        print(hidden_size2)

        self.fc1 = nn.Linear(hidden_size1, hidden_size2, bias=False)
        self.fc2 = nn.Linear(hidden_size2, output_size, bias=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.flatten = nn.Flatten()

        self.p_scale = float(img_size - 1)
        self.w_M = w_M
        self.w_m = w_m
        self.w_scale = w_M - w_m
        self.scale_W = torch.tensor([[[self.p_scale, self.p_scale, self.w_scale]]])
        self.scale_b = torch.tensor([[[0., 0., w_m]]])
        
    def forward(self, data):
        x = self.relu(self.conv1(data)) # conv1 -> relu
        x = self.pool(x) # pool
        x = self.relu(self.conv2(x)) # conv2 -> relu
        x = self.pool(x) # pool
        # x = x.view(x.size(0), -1) # reshape to use linear function
        x = self.flatten(x.permute(0,2,3,1))
        x = self.sigmoid(self.fc1(x)) # fc1 -> sigmoid
        y = self.sigmoid(self.fc2(x)) # fc1 -> sigmoid
        raw = y.reshape(-1, self.n_point, 3)
        out = raw*self.scale_W + self.scale_b
        return out

class Generator4(nn.Module):# 太さ対応生成器(3) 中でソートしてみた。性能が下がるだけか？
    def __init__(self, img_size, n_point, w_m=0.25, w_M=4.):
        super().__init__()
        self.n_point = n_point
        self.img_size = img_size
        
        self.flatten = nn.Flatten()
        
        input_size = img_size**2
        output_size = 3*n_point
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, stride=2)

        conved = (img_size - 4)//2
        conved = (conved - 4)//2

        self.conved = conved

        hidden_size1 = 3*conved**2
        #print(hidden_size1)
        
        hidden_size2 = 3*((conved**2 + output_size)//2)
        #print(hidden_size2)

        self.fc1 = nn.Linear(hidden_size1, hidden_size2, bias=False)
        self.fc2 = nn.Linear(hidden_size2, output_size, bias=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.flatten = nn.Flatten()

        self.p_scale = float(img_size - 1)
        self.w_M = w_M
        self.w_m = w_m
        self.w_scale = w_M - w_m
        self.scale_W = torch.tensor([[[self.p_scale, self.p_scale, self.w_scale]]])
        self.scale_b = torch.tensor([[[0., 0., w_m]]])
        
    def forward(self, data):
        N = data.size()[0]

        x = self.relu(self.conv1(data)) # conv1 -> relu
        x = self.pool(x) # pool
        x = self.relu(self.conv2(x)) # conv2 -> relu
        x = self.pool(x) # pool
        # x = x.view(x.size(0), -1) # reshape to use linear function

        xf = x.reshape(N, 3, self.conved**2)
        val = xf[:, 0]
        idx = torch.argsort(val, dim = -1)
        sorted1 = torch.gather(xf, -1, idx.unsqueeze(1).repeat(1,3,1))
        x1 = self.flatten(sorted1.permute(0, 2, 1))

        # x = self.flatten(x.permute(0,2,3,1))
        x1 = self.sigmoid(self.fc1(x1)) # fc1 -> sigmoid

        y = self.sigmoid(self.fc2(x1)) # fc1 -> sigmoid
        raw = y.reshape(-1, self.n_point, 3)
        out = raw*self.scale_W + self.scale_b
        return out

class Generator5(nn.Module):# 太さ対応生成器(4) 中でソート part2
    def __init__(self, img_size, n_point, w_m=0.25, w_M=4.):
        super().__init__()
        self.n_point = n_point
        self.img_size = img_size
        
        self.flatten = nn.Flatten()
        
        input_size = img_size**2
        output_size = 3*n_point
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, padding=0) # 0チャンネル目で残りをソートする
        self.pool = nn.MaxPool2d(2, stride=2)

        conved = (img_size - 4)//2
        conved = (conved - 4)//2

        self.conved = conved

        hidden_size1 = 3*conved**2
        #print(hidden_size1)
        
        hidden_size2 = 3*((conved**2 + output_size)//2)
        #print(hidden_size2)

        self.fc1 = nn.Linear(hidden_size1, hidden_size2, bias=False)
        self.fc2 = nn.Linear(hidden_size2, output_size, bias=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.flatten = nn.Flatten()

        self.p_scale = float(img_size - 1)
        self.w_M = w_M
        self.w_m = w_m
        self.w_scale = w_M - w_m
        self.scale_W = torch.tensor([[[self.p_scale, self.p_scale, self.w_scale]]])
        self.scale_b = torch.tensor([[[0., 0., w_m]]])
        
    def forward(self, data):
        N = data.size()[0]

        x = self.relu(self.conv1(data)) # conv1 -> relu
        x = self.pool(x) # pool
        x = self.relu(self.conv2(x)) # conv2 -> relu
        x = self.pool(x) # pool
        # x = x.view(x.size(0), -1) # reshape to use linear function

        xf = x.reshape(N, 4, self.conved**2)
        val = xf[:, 0]
        obj = xf[:, 1:]
        idx = torch.argsort(val, dim = -1)
        sorted1 = torch.gather(obj, -1, idx.unsqueeze(1).repeat(1,3,1))
        x1 = self.flatten(sorted1.permute(0, 2, 1))

        # x = self.flatten(x.permute(0,2,3,1))
        x1 = self.sigmoid(self.fc1(x1)) # fc1 -> sigmoid

        y = self.sigmoid(self.fc2(x1)) # fc1 -> sigmoid
        raw = y.reshape(-1, self.n_point, 3)
        out = raw*self.scale_W + self.scale_b
        return out

class Generator6(nn.Module):# 太さ対応生成器(5) 設計変更
    def __init__(self, img_size, n_point, w_m=0.25, w_M=4.):
        super().__init__()
        self.n_point = n_point
        self.img_size = img_size
        
        self.flatten = nn.Flatten()
        
        #input_size = img_size**2
        output_size = 3*n_point
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, stride=2)

        conved = (img_size)//2
        conved = (conved)//2
        conved = (conved)//2
        hidden_size1 = 3*conved**2
        # print(hidden_size1)
        
        hidden_size2 = 3*((conved**2 + n_point)//2)
        # print(hidden_size2)

        self.fc1 = nn.Linear(hidden_size1, hidden_size2, bias=False)
        self.fc2 = nn.Linear(hidden_size2, output_size, bias=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.flatten = nn.Flatten()

        self.p_scale = float(img_size - 1)
        self.w_M = w_M
        self.w_m = w_m
        self.w_scale = w_M - w_m
        self.scale_W = nn.Parameter(torch.tensor([[[self.p_scale, self.p_scale, self.w_scale]]]), requires_grad=False)
        self.scale_b = nn.Parameter(torch.tensor([[[0., 0., w_m]]]), requires_grad=False)

        self.bn2d = nn.BatchNorm2d(3)
        self.bn1d = nn.BatchNorm1d(hidden_size2)
        
    def forward(self, data):
        x = self.relu(self.conv1(data)) # conv1 -> relu
        x = self.pool(x) # pool
        x = self.bn2d(x) # batch norm
        x = self.relu(self.conv2(x)) # conv2 -> relu
        x = self.pool(x) # pool
        x = self.relu(self.conv3(x)) # conv3 -> relu
        x = self.pool(x) # pool
        x = self.flatten(x.permute(0,2,3,1))
        x = self.sigmoid(self.fc1(x)) # fc1 -> sigmoid
        x = self.bn1d(x) # batch norm
        y = self.sigmoid(self.fc2(x)) # fc2 -> sigmoid
        raw = y.reshape(-1, self.n_point, 3)
        out = raw*self.scale_W + self.scale_b
        return out

class Generator7(nn.Module):# 太さ対応生成器(6) 設計変更
    def __init__(self, img_size, n_point, w_m=0.25, w_M=4.):
        super().__init__()
        self.n_point = n_point
        self.img_size = img_size
        
        self.flatten = nn.Flatten()
        
        #input_size = img_size**2
        output_size = 3*n_point
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, stride=2)

        conved = (img_size)//2
        conved = (conved)//2
        conved = (conved)//2
        hidden_size1 = 3*conved**2
        # print(hidden_size1)
        
        hidden_size2 = 3*((conved**2 + n_point)//2)
        # print(hidden_size2)

        self.fc1 = nn.Linear(hidden_size1, hidden_size2, bias=False)
        self.fc2 = nn.Linear(hidden_size2, output_size, bias=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.flatten = nn.Flatten()

        self.p_scale = float(img_size - 1)
        self.w_M = w_M
        self.w_m = w_m
        self.w_scale = w_M - w_m
        self.scale_W = nn.Parameter(torch.tensor([[[self.p_scale, self.p_scale, self.w_scale]]]), requires_grad=False)
        self.scale_b = nn.Parameter(torch.tensor([[[0., 0., w_m]]]), requires_grad=False)
        
    def forward(self, data):
        x = self.relu(self.conv1(data)) # conv1 -> relu
        x = self.pool(x) # pool
        x = self.relu(self.conv2(x)) # conv2 -> relu
        x = self.pool(x) # pool
        x = self.relu(self.conv3(x)) # conv3 -> relu
        x = self.pool(x) # pool
        x = self.flatten(x.permute(0,2,3,1))
        x = self.sigmoid(self.fc1(x)) # fc1 -> sigmoid
        y = self.sigmoid(self.fc2(x)) # fc2 -> sigmoid
        raw = y.reshape(-1, self.n_point, 3)
        out = raw*self.scale_W + self.scale_b
        return out

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
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

    # def forward(self, token_embedding: torch.tensor) -> torch.tensor:
    #     token_embedding = torch.transpose(token_embedding, 0, 1)
    #     return torch.transpose(self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :]), 0, 1)

class Generator8(nn.Module):# 太さ対応生成器(7) 点の数までチャンネル数を増やし，transformerへ img_size=48, n_point=16
    def __init__(self, w_m=0., w_M=4.):
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

        # self.linear_mapping = nn.Sequential(
        #     nn.Linear(in_features=36, out_features=20), nn.Sigmoid(),
        #     nn.Linear(in_features=20, out_features=3)
        # )

        self.linear_mapping = nn.Linear(in_features=36, out_features=3)

        self.sigmoid = nn.Sigmoid()

        self.flatten = nn.Flatten()

        self.p_scale = float(self.img_size - 1)
        self.w_M = w_M
        self.w_m = w_m
        self.w_scale = w_M - w_m
        self.scale_W = nn.Parameter(torch.tensor([[[self.p_scale, self.p_scale, self.w_scale]]]), requires_grad=False)
        self.scale_b = nn.Parameter(torch.tensor([[[0., 0., w_m]]]), requires_grad=False)
        
    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], 16, 36)
        # print(x.shape)
        x = x.permute(1,0,2)
        # print(x.shape)
        x = self.positional_encoder(x)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        y = self.sigmoid(self.linear_mapping(x))
        # print(y.shape)
        raw = y.permute(1,0,2)
        out = raw*self.scale_W + self.scale_b
        return out

class Generator9(nn.Module):# 太さ対応生成器(7.1) 絵が描けないか？... img_size=128, n_point=32
    def __init__(self, w_m=0., w_M=4., dim_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.img_size = 128
        self.n_point = 32
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, padding=2),nn.ReLU(),nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, padding=2),nn.ReLU(),nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2),nn.ReLU(),nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),nn.ReLU(),nn.MaxPool2d(2, stride=2)
        )

        self.positional_encoder = PositionalEncoding(dim_model=dim_model, max_len=40, dropout_p=0.1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(dim_model))

        # self.linear_mapping = nn.Sequential(
        #     nn.Linear(in_features=36, out_features=20), nn.Sigmoid(),
        #     nn.Linear(in_features=20, out_features=3)
        # )

        self.linear_mapping = nn.Linear(in_features=dim_model, out_features=3)

        self.sigmoid = nn.Sigmoid()

        self.p_scale = float(self.img_size - 1)
        self.w_M = w_M
        self.w_m = w_m
        self.w_scale = w_M - w_m
        self.scale_W = nn.Parameter(torch.tensor([[[self.p_scale, self.p_scale, self.w_scale]]]), requires_grad=False)
        self.scale_b = nn.Parameter(torch.tensor([[[0., 0., w_m]]]), requires_grad=False)
        
    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], 32, 64)
        # print(x.shape)
        x = x.permute(1,0,2)
        # print(x.shape)
        x = self.positional_encoder(x)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        y = self.sigmoid(self.linear_mapping(x))
        # print(y.shape)
        raw = y.permute(1,0,2)
        out = raw*self.scale_W + self.scale_b
        return out

class Generator10(nn.Module):# 太さ対応生成器(7.2) 絵が描けないか？... img_size=48, n_point=32
    def __init__(self, w_m=0., w_M=4.):
        super().__init__()
        self.img_size = 48
        self.n_point = 32
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, padding=2),nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, padding=2),nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2),nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.positional_encoder = PositionalEncoding(dim_model=36, max_len=40, dropout_p=0.1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=36, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=3, norm=nn.LayerNorm(36))

        # self.linear_mapping = nn.Sequential(
        #     nn.Linear(in_features=36, out_features=20), nn.Sigmoid(),
        #     nn.Linear(in_features=20, out_features=3)
        # )

        self.linear_mapping = nn.Linear(in_features=36, out_features=3)

        self.sigmoid = nn.Sigmoid()

        self.flatten = nn.Flatten()

        self.p_scale = float(self.img_size - 1)
        self.w_M = w_M
        self.w_m = w_m
        self.w_scale = w_M - w_m
        self.scale_W = nn.Parameter(torch.tensor([[[self.p_scale, self.p_scale, self.w_scale]]]), requires_grad=False)
        self.scale_b = nn.Parameter(torch.tensor([[[0., 0., w_m]]]), requires_grad=False)
        
    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], 32, 36)
        # print(x.shape)
        x = x.permute(1,0,2)
        # print(x.shape)
        x = self.positional_encoder(x)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        y = self.sigmoid(self.linear_mapping(x))
        # print(y.shape)
        raw = y.permute(1,0,2)
        out = raw*self.scale_W + self.scale_b
        return out

class Generator8_2(nn.Module):# 太さ対応生成器(7) 点の数までチャンネル数を増やし，transformerへ img_size=48, n_point=16, mask付き
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
        
        #self.save_x = None
        
    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], 16, 36)
        # print(x.shape)
        x = x.permute(1,0,2)
        #self.save_x = x.clone().detach()
        x = self.positional_encoder(x)
        #print(x)

        x = self.encoder(x, mask=self.mask)
        # print(x.shape)
        y = self.sigmoid(self.linear_mapping(x))
        # print(y.shape)
        raw = y.permute(1,0,2)
        out = raw*self.scale_W + self.scale_b
        return out
