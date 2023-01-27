import torch

import torchvision
from torchvision import datasets,transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from tqdm import tqdm

import sys, os
sys.path.append('..')

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from hiragana_dataset import make_hiragana_dataset
import glob

def torch_fix_seed():
    """
    ランダム値の生成を固定する。
    """
    seed = 42
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

# 1文字に専念
def train1(gen, disc, inputs, labels, iteration=2000, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using: {device}')

    gen = gen.to(device)
    disc = disc.to(device)
    
    optimizer=optim.Adadelta(gen.parameters(), lr=lr)
    
    # points_log = []
    # plot_log = []
    losses = []

    gen = gen.train()

    for i in range(iteration):
        optimizer.zero_grad()
        
        points = gen(inputs)
        
        loss = disc(points, labels)
        
        loss.backward()
    
        optimizer.step()

        losses.append(loss.cpu().detach().numpy())
        if i%100 == 0:
            print(f'iter: {i}, loss: {loss}')
            # points_log.append(points)
            # plot_log.append(disc2.p_map.cpu())
    
    errors = disc.error_log
    distances = disc.distance_log
    #distributions = disc.distribution_log
            
    # return points_log, plot_log, losses, errors, distances, distributions
    # return losses
    return losses, errors, distances

# 1文字に専念:ノイズ
def train1_by_noise(gen, disc, tgt, iteration=2000, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using: {device}')

    gen = gen.to(device)
    disc = disc.to(device)
    
    optimizer=optim.Adam(gen.parameters(), lr=lr)
    # optimizer=optim.Adadelta(gen.parameters(), lr=lr)
    
    # points_log = []
    # plot_log = []
    losses = []

    gen = gen.train()

    for i in range(iteration):
        noise1, noise2 = torch.randn(2,1,48,48)
        inputs = torch.stack([tgt, noise1, noise2], dim=1)
        inputs.to(device)

        optimizer.zero_grad()
        
        points = gen(inputs)
        
        loss = disc(points, tgt)
        
        loss.backward()
    
        optimizer.step()

        losses.append(loss.cpu().detach().numpy())
        if i%100 == 0:
            print(f'iter: {i}, loss: {loss}')
            # points_log.append(points)
            # plot_log.append(disc2.p_map.cpu())
    
    errors = disc.error_log
    distances = disc.distance_log
    #distributions = disc.distribution_log
            
    # return points_log, plot_log, losses, errors, distances, distributions
    # return losses
    return losses, errors, distances

# 1文字に専念:回転
def train1_rotation(gen, disc, inputs, labels, iteration=2000, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using: {device}')

    gen = gen.to(device)
    disc = disc.to(device)
    
    optimizer=optim.Adadelta(gen.parameters(), lr=lr)

    transform = transforms.Compose(
        [transforms.RandomRotation(degrees=[-180, 180]),]
    )
    
    losses = []

    gen = gen.train()

    for i in range(iteration):
        rotated = transform(inputs)
        optimizer.zero_grad()
        
        points = gen(rotated)
        
        loss = disc(points, labels)
        
        loss.backward()
    
        optimizer.step()

        losses.append(loss.cpu().detach().numpy())
        if i%100 == 0:
            print(f'iter: {i}, loss: {loss}')
            # points_log.append(points)
            # plot_log.append(disc2.p_map.cpu())
    
    errors = disc.error_log
    distances = disc.distance_log
    return losses, errors, distances

# 1文字に専念:scheduler付き
def train1_schedule(gen, disc, inputs, labels, iteration=2000, start_factor=1, end_factor=0.001, total_iters=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using: {device}')

    gen = gen.to(device)
    disc = disc.to(device)
    
    optimizer=optim.Adadelta(gen.parameters(), lr=start_factor)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)
    
    # points_log = []
    # plot_log = []
    losses = []

    gen = gen.train()

    for i in range(iteration):
        optimizer.zero_grad()
        
        points = gen(inputs)
        
        loss = disc(points, labels)
        
        loss.backward()
    
        optimizer.step()

        losses.append(loss.cpu().detach().numpy())
        if i%100 == 0:
            print(f'iter: {i}, loss: {loss}')
            scheduler.step()
    
    errors = disc.error_log
    distances = disc.distance_log
    #distributions = disc.distribution_log
            
    # return points_log, plot_log, losses, errors, distances, distributions
    # return losses
    return losses, errors, distances


# 1文字に専念:scheduler付き, labelに直線ノイズを加える
import random

def train1_schedule2(gen, disc, inputs, labels, iteration=2000, start_factor=1, end_factor=0.001, total_iters=20, add_line_p=0.25, d=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using: {device}')

    def add_random_line(x, d):
        x_noise = x.clone()
        ix = random.randrange(96)
        if ix > 47:
            ix = list(range(48))
            iy = random.randrange(96)
            if iy > 47:
                if iy%2 == 0:
                    iy = list(range(48))
                else:
                    iy = list(reversed(range(48)))
        else:
            iy = list(range(48))

        x_noise[:, ix, iy] = d
        return x_noise

    gen = gen.to(device)
    disc = disc.to(device)
    
    optimizer=optim.Adadelta(gen.parameters(), lr=start_factor)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)
    
    # points_log = []
    # plot_log = []
    losses = []

    gen = gen.train()

    for i in range(iteration):
        add_line = random.random()
        if add_line < add_line_p:
            labels_ = add_random_line(labels, d)
        else:
            labels_ = labels
        
        optimizer.zero_grad()
        
        points = gen(inputs)
        
        loss = disc(points, labels_)
        
        loss.backward()
    
        optimizer.step()

        losses.append(loss.cpu().detach().numpy())
        if i%100 == 0:
            print(f'iter: {i}, loss: {loss}')
            scheduler.step()
    
    errors = disc.error_log
    distances = disc.distance_log
    #distributions = disc.distribution_log
            
    # return points_log, plot_log, losses, errors, distances, distributions
    # return losses
    return losses, errors, distances

# 1文字に専念:scheduler付き, labelにノイズを加える
def train1_schedule3(gen, disc, inputs, labels, iteration=2000, start_factor=1, end_factor=0.001, total_iters=20, p=0.25, d=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using: {device}')

    def add_noise(x, p, d):
        flatten = nn.Flatten()
        x_noise = x.clone()
        original_shape = x_noise.shape
        x_noise = flatten(x_noise)
        length = x_noise.shape[-1]
        tgt_num = int(length*p)
        tgt = np.random.choice(length, tgt_num, replace=False)
        x_noise[:, tgt] = d
        x_noise = x_noise.reshape(*original_shape)
        return x_noise

    gen = gen.to(device)
    disc = disc.to(device)
    
    optimizer=optim.Adadelta(gen.parameters(), lr=start_factor)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)
    
    # points_log = []
    # plot_log = []
    losses = []

    gen = gen.train()

    for i in range(iteration):
        labels_ = add_noise(labels, p, d)
        
        optimizer.zero_grad()
        
        points = gen(inputs)
        
        loss = disc(points, labels_)
        
        loss.backward()
    
        optimizer.step()

        losses.append(loss.cpu().detach().numpy())
        if i%100 == 0:
            print(f'iter: {i}, loss: {loss}')
            scheduler.step()
    
    errors = disc.error_log
    distances = disc.distance_log
    #distributions = disc.distribution_log
            
    # return points_log, plot_log, losses, errors, distances, distributions
    # return losses
    return losses, errors, distances

# 1文字に専念:scheduler付き, 加えるノイズを段階的に下げていく
def train1_schedule4(gen, disc, inputs, labels, iteration=2000, start_factor=1, end_factor=0.001, total_iters=20, start_p=0.5, alpha=400, beta=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using: {device}')

    def add_noise(x, p):
        flatten = nn.Flatten()
        x_noise = x.clone()
        original_shape = x_noise.shape
        x_noise = flatten(x_noise)
        length = x_noise.shape[-1]
        tgt_num = int(length*p)
        tgt = np.random.choice(length, tgt_num, replace=False)
        x_noise[:, tgt] = 1.0
        x_noise = x_noise.reshape(*original_shape)
        return x_noise

    gen = gen.to(device)
    disc = disc.to(device)
    
    optimizer=optim.Adadelta(gen.parameters(), lr=start_factor)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)
    
    # points_log = []
    # plot_log = []
    losses = []

    gen = gen.train()

    p = start_p

    for i in range(iteration):
        p = start_p*np.exp(-i/alpha)*(np.cos(np.pi*i/beta) + 1) #振動させる

        labels_ = add_noise(labels, p)
        
        optimizer.zero_grad()
        
        points = gen(inputs)
        
        loss = disc(points, labels_)
        
        loss.backward()
    
        optimizer.step()

        losses.append(loss.cpu().detach().numpy())
        if i%100 == 0:
            print(f'iter: {i}, loss: {loss}')
            scheduler.step()
    
    errors = disc.error_log
    distances = disc.distance_log
    #distributions = disc.distribution_log
            
    # return points_log, plot_log, losses, errors, distances, distributions
    # return losses
    return losses, errors, distances

# 1文字に専念:scheduler付き, 分割あり->ノイズ付加，段階的に下げていく．分割なし->ノイズ少なく
from dfs import how_many_island

def train1_schedule5(gen, disc, inputs, labels,
    iteration=2000, start_factor=1, end_factor=0.001, total_iters=20, 
    start_p=0.5, alpha=400, beta=50,
    folder="model"):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using: {device}')

    torch_fix_seed()

    def add_noise(x, p):
        flatten = nn.Flatten()
        x_noise = x.clone()
        original_shape = x_noise.shape
        x_noise = flatten(x_noise)
        length = x_noise.shape[-1]
        tgt_num = int(length*p)
        tgt = np.random.choice(length, tgt_num, replace=False)
        x_noise[:, tgt] = 1.0
        x_noise = x_noise.reshape(*original_shape)
        return x_noise
    
    gen = gen.to(device)
    disc = disc.to(device)
    
    optimizer=optim.Adadelta(gen.parameters(), lr=start_factor)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)
    
    losses = []

    gen = gen.train()

    split = (how_many_island(labels.cpu().detach().tolist()[0]) > 1) #!
    print("split:", split)

    if not split:
        start_p = start_p/4
        alpha = alpha/4
        beta = beta/2
    
    min_loss = 999999999.9
    avg_loss = 0.
    for i in range(iteration):
        p = start_p*np.exp(-i/alpha)*(np.cos(np.pi*i/beta) + 1) #振動させる
        labels_ = add_noise(labels, p)
        
        optimizer.zero_grad()
        
        points = gen(inputs)
        
        loss = disc(points, labels_)
        
        loss.backward()
    
        optimizer.step()

        losses.append(loss.cpu().detach().numpy())
        avg_loss += loss.detach().item()

        if (i+1)%100 == 0:
            avg_loss = avg_loss/100
            print(f'iter: {i+1}, loss: {avg_loss}')
            scheduler.step()

            if avg_loss < min_loss:
                min_loss = avg_loss
                model_list = glob.glob("../model/"+folder+"/*")
                for prev_model in model_list:
                    os.remove(prev_model)
                torch.save(gen.state_dict(), "../model/"+folder+"/best.pt")
            
            avg_loss = 0
    
    errors = disc.error_log
    distances = disc.distance_log

    return losses, errors, distances


# 分裂の判別なし
def train1_schedule5_2(gen, disc, inputs, labels,
    iteration=2000, start_factor=1, end_factor=0.001, total_iters=20, 
    start_p=0.5, alpha=400, beta=50,
    folder="model"):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using: {device}')

    torch_fix_seed()

    def add_noise(x, p):
        flatten = nn.Flatten()
        x_noise = x.clone()
        original_shape = x_noise.shape
        x_noise = flatten(x_noise)
        length = x_noise.shape[-1]
        tgt_num = int(length*p)
        tgt = np.random.choice(length, tgt_num, replace=False)
        x_noise[:, tgt] = 1.0
        x_noise = x_noise.reshape(*original_shape)
        return x_noise
    
    gen = gen.to(device)
    disc = disc.to(device)
    
    optimizer=optim.Adadelta(gen.parameters(), lr=start_factor)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)
    
    losses = []

    gen = gen.train()
    
    min_loss = 999999999.9
    avg_loss = 0.
    for i in range(iteration):
        p = start_p*np.exp(-i/alpha)*(np.cos(np.pi*i/beta) + 1) #振動させる
        labels_ = add_noise(labels, p)
        
        optimizer.zero_grad()
        
        points = gen(inputs)
        
        loss = disc(points, labels_)
        
        loss.backward()
    
        optimizer.step()

        losses.append(loss.cpu().detach().numpy())
        avg_loss += loss.detach().item()

        if (i+1)%100 == 0:
            avg_loss = avg_loss/100
            print(f'iter: {i+1}, loss: {avg_loss}')
            scheduler.step()

            if avg_loss < min_loss:
                min_loss = avg_loss
                model_list = glob.glob("../model/"+folder+"/*")
                for prev_model in model_list:
                    os.remove(prev_model)
                torch.save(gen.state_dict(), "../model/"+folder+"/best.pt")
            
            avg_loss = 0
    
    errors = disc.error_log
    distances = disc.distance_log

    return losses, errors, distances


def train1_schedule6(gen, disc, inputs, labels, iteration=2000, start_factor=1, end_factor=0.001, total_iters=20, start_p=0.5, alpha=400, beta=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using: {device}')

    torch_fix_seed()

    def add_noise(x, p):
        flatten = nn.Flatten()
        x_noise = x.clone()
        original_shape = x_noise.shape
        x_noise = flatten(x_noise)
        length = x_noise.shape[-1]
        tgt_num = int(length*p)
        tgt = np.random.choice(length, tgt_num, replace=False)
        x_noise[:, tgt] = 1.0
        x_noise = x_noise.reshape(*original_shape)
        return x_noise
    
    gen = gen.to(device)
    disc = disc.to(device)
    inputs.to(device)
    labels.to(device)
    
    optimizer=optim.Adadelta(gen.parameters(), lr=start_factor)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)
    
    losses = []

    gen = gen.train()

    for i in range(iteration):
        p = start_p*np.exp(-i/alpha)*(np.cos(np.pi*i/beta) + 1) #振動させる
        labels_ = add_noise(labels, p)
        
        optimizer.zero_grad()
        
        points = gen(inputs)
        
        loss = disc(points, labels_)
        
        loss.backward()
    
        optimizer.step()

        losses.append(loss.cpu().detach().numpy())
        if i%100 == 0:
            print(f'iter: {i}, loss: {loss}')
            scheduler.step()
    
    errors = disc.error_log
    distances = disc.distance_log
    
    return losses, errors, distances

# 汎用化の試み
def train_loop(gen, disc, opt, train_dataloader, device, augment=True):
    gen.train()
    epoch_loss = 0
    transform = transforms.Compose(
        [transforms.RandomRotation(degrees=[-180, 180]),]
    )
    for src_batch, tgt_batch in tqdm(train_dataloader):
        if augment:
            src_batch = transform(src_batch) #データ拡張追加

        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)

        # 生成
        points = gen(src_batch)
        # 評価
        loss = disc(points, tgt_batch)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.detach().item()
        
    return epoch_loss / len(train_dataloader)

def valid_loop(gen, disc, val_dataloader, device):
    gen.eval()
    total_loss = 0
    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(val_dataloader):
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            # 生成
            points = gen(src_batch)
            # 評価
            loss = disc(points, tgt_batch)
            total_loss += loss.detach().item()
        
    return total_loss / len(val_dataloader)

def train(gen, disc, DATADIR, CATEGORIES=None, batch_size=100, num_epochs=30, lr=0.001, augment=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'using: {device}')

    train_dataset, valid_dataset = make_hiragana_dataset(DATADIR, CATEGORIES)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    gen = gen.to(device)
    disc = disc.to(device)

    # opt=optim.Adadelta(gen.parameters(), lr=lr)
    opt=optim.Adam(gen.parameters(), lr=lr)

    train_loss_list = []
    validation_loss_list = []
    # min_validation_loss = float('inf')

    for epoch in range(num_epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        print("Training")
        train_loss = train_loop(gen, disc, opt, train_dataloader, device, augment)
        train_loss_list += [train_loss]
        print("Validating")
        validation_loss = valid_loop(gen, disc, val_dataloader, device)
        validation_loss_list += [validation_loss]
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        # if validation_loss < min_validation_loss:
        #     min_validation_loss = validation_loss
        #     model_list = glob.glob("../model/*")
        #     for prev_model in model_list:
        #         os.remove(prev_model)
        #     torch.save(model.state_dict(), "../model/best.pt")

    return train_loss_list, validation_loss_list

