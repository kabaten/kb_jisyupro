import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import glob
import random
import sys, os
sys.path.append('..')

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

def train(gen, disc, inputs, labels,
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
        p = start_p*np.exp(-i/alpha)*(np.cos(np.pi*i/beta) + 1) # 減衰振動させる
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