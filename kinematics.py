import numpy as np
import matplotlib.pyplot as plt

class MyLink:
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2
    
    def fk(self, th):
        th1, th2 = th
        e1 = np.stack([np.cos(th1),
                       np.sin(th1)])
        e2 = np.stack([-np.cos(th2), 
                       np.sin(th2)])
        p = self.l1*e1 - self.l2*e2
        # mask = (th1 + th2 <= np.pi)
        # p = p[:, mask.squeeze()]
        return p
    
    def ik(self, p):
        x, y = p
        p_norm = np.linalg.norm(p, ord=2, axis=0)
        alpha = np.arctan2(y, x)
        
        th1 = np.arccos((p_norm**2 + self.l1**2 - self.l2**2)/(2*self.l1*p_norm)) + alpha
        th2 = np.arccos((p_norm**2 + self.l2**2 - self.l1**2)/(2*self.l2*p_norm)) - alpha
        
        th = np.stack[th1, th2]
        return th