import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import torch

"""
出力行列をスプライン補完する関数
"""
def spline(x,y,num,deg):
    tck,u = interpolate.splprep([x,y],k=deg,s=0) 
    u = np.linspace(0,1,num=num,endpoint=True) 
    spline = interpolate.splev(u,tck)
    return spline[0],spline[1]

def complement(points, num, deg):
    x, y = points
    x, y = spline(x,y,num,deg)
    return np.array([x, y])

if __name__ == '__main__':
    points = torch.randint(0, 48, [5, 2]).to(torch.float)
    print(points)
    points = points.detach().numpy().T
    print(points)
    px, py = points
    x, y = complement(points,100,3)
    plt.scatter(px, py)
    plt.plot(x, y)
    plt.show()
