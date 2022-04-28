import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import os
import time

from tqdm import tqdm

from ..metrics import MSE


def get_grad(model, z,device):
		inputs=Variable(torch.tensor([z[0][0],z[1][0]]), requires_grad = True).to(device)
		out=model(inputs.float())
		dH=torch.autograd.grad(out, inputs, grad_outputs=out.data.new(out.shape).fill_(1),create_graph=True)[0]
		return dH.detach().cpu().numpy()[1], -dH.detach().cpu().numpy()[0] # negative dH/dq is dp/dt
	
def LeapfrogIntNN(z,h,model,device):
## classical Leapfrog scheme for force field f
# can compute multiple initial values simultanously, z[k]=list of k-component of all initial values
	dim = int(len(z)/2)
	z[dim:] = z[dim:]+h/2*get_grad(model, z,device)[1]
	z[:dim] = z[:dim]+h*get_grad(model, z,device)[0]
	z[dim:] = z[dim:]+h/2*get_grad(model, z,device)[1]
	return z
  
def LeapfrogTrajectoryNN(traj_len,start,h,model,n_h = 100):
  h_gen = h/n_h
  x, final = start.copy(), start.copy()
  for i in range(traj_len):
    start=np.hstack((start,x))
    for j in range(0,int(n_h)):
      x=LeapfrogIntNN(x,h_gen,model,device)
    final=np.hstack((final,x))
  return start[:,1:],final[:,1:]
