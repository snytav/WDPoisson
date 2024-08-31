import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from surface import plot_density_surface

class WDPoisson(nn.Module):
    def __init__(self,Nx,Ny,Lx,Ly,N):
        super(WDPoisson,self).__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.N  = N
        fc1 = nn.Linear(self.Nx*self.Ny*2,self.N)
        self.fc1 = fc1
        self.fc2 = nn.Linear(self.N,self.Nx*self.Ny*2)

    def forward(self,x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return 0.0
    def forward_simulate(self,x):
        X = x[:,:,0]
        Y = x[:,:,1]
        y = torch.sin(2*torch.pi*X)
        return y

if __name__ == '__main__':
    N = 10
    x = np.linspace(0,1.0,N)
    y = np.linspace(0,1.0,N)
    X,Y = np.meshgrid(x,y)
    X = torch.from_numpy(X)#.reshape(X.shape[0]*X.shape[0])
    Y = torch.from_numpy(Y)#.reshape(Y.shape[0]*Y.shape[0])
    X = X.reshape(X.shape[0], X.shape[1], 1)
    Y = Y.reshape(Y.shape[0], Y.shape[1], 1)
    XY = torch.cat((X, Y), dim=2)

    #XY = torch.cat((X,Y),1)
    model = WDPoisson(X.shape[0],X.shape[1],x[-1],y[-1],10)
    y = model.forward_simulate(XY)
    plot_density_surface(y.T,(int(X.shape[0]),int(X.shape[1])),
                         x[1]-x[0],'sin(X)')
    plt.show(block=True)
    qq = 0



