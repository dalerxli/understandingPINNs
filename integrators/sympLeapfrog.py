import numpy as np
import os
import time

from tqdm import tqdm

from ..metrics import MSE

def LeapfrogIntNN(z,h,model):
## classical Leapfrog scheme for force field f
# can compute multiple initial values simultanously, z[k]=list of k-component of all initial values
	dim = int(len(z)/2)
	z[dim:] = z[dim:]+h/2*torch.squeeze(model(torch.tensor(z).transpose(1,0).float()),0).detach().numpy().transpose()[1]
	z[:dim] = z[:dim]+h*torch.squeeze(model(torch.tensor(z).transpose(1,0).float()),0).detach().numpy().transpose()[0]
	z[dim:] = z[dim:]+h/2*torch.squeeze(model(torch.tensor(z).transpose(1,0).float()),0).detach().numpy().transpose()[1]
	return z
  
def LeapfrogTrajectoryNN(traj_len,start,h,model,n_h = 100):
  h_gen = h/n_h
  x, final = start.copy(), start.copy()
  for i in range(traj_len):
    start=np.hstack((start,x))
    for j in range(0,int(n_h)):
      x=LeapfrogIntNN(x,h_gen,model)
    final=np.hstack((final,x))
  return start[:,1:],final[:,1:]
