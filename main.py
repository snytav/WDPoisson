import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from surface import plot_density_surface
from torch.autograd.functional import jacobian,hessian

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
    N = 4
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
    f = lambda x, y: x ** 2 + y ** 2
    f44 = f(torch.ones((4, 4)), 2 * torch.ones((4, 4)))
    j44 = jacobian(f, inputs=(torch.ones((4, 4)), 2 * torch.ones((4, 4))))
    vec_df_dX = j44[0] # X = (torch.ones((2, 2))     # vector of derivatives of f wrt all X matric components
    vec_df_dY = j44[1] # Y = 4 * torch.ones((4, 4))
    df_dX = torch.sum(torch.flatten(vec_df_dX,start_dim=0,end_dim=1),dim=0)

    plot_density_surface(df_dX.T, (int(X.shape[0]), int(X.shape[1])),
                         x[1] - x[0], 'sin(X)')
    plt.show(block=True)
    df_dY = torch.sum(torch.flatten(vec_df_dY, start_dim=0, end_dim=1), dim=0)
    plot_density_surface(df_dY.T, (int(X.shape[0]), int(X.shape[1])),
                         x[1] - x[0], 'sin(X)')
    plt.show(block=True)

    # in order to get the plain df_dX  a sum of vec_df_dX components must be performed with inidices corresponding
    # the position of nonzero element in matrix
    # df_dX = np.zeros_like()
    # for (i,j)
    qq = 0
