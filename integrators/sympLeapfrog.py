import numpy as np
import os
import time

from tqdm import tqdm

from ..metrics import MSE

def LeapfrogNNH_autograd(z,h,model):
## classical Leapfrog scheme for force field f
# can compute multiple initial values simultanously, z[k]=list of k-component of all initial values
	dim = int(len(z)/2)
	z[dim:] = z[dim:]+h/2*torch.squeeze(model(torch.tensor(z).transpose(1,0).float()),0).detach().numpy().transpose()[1]
	z[:dim] = z[:dim]+h*torch.squeeze(model(torch.tensor(z).transpose(1,0).float()),0).detach().numpy().transpose()[0]
	z[dim:] = z[dim:]+h/2*torch.squeeze(model(torch.tensor(z).transpose(1,0).float()),0).detach().numpy().transpose()[1]
	return z
  
def gen_one_trajNNH_autograd(traj_len,start,h,model,n_h = 100):
  h_gen = h/n_h
  x, final = start.copy(), start.copy()
  for i in range(traj_len):
    start=np.hstack((start,x))
    for j in range(0,int(n_h)):
      x=LeapfrogNNH_autograd(x,h_gen,model)
    final=np.hstack((final,x))
  return start[:,1:],final[:,1:]
