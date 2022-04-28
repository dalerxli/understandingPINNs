import numpy as np
import os
import time

from tqdm import tqdm

from ..metrics import MSE

def naiveIntNN(z,h,net):
		dim = int(len(z)/2)
		q=z[:dim]
		p=z[dim:]		
		dH = torch.squeeze(net(torch.tensor(z).float()),0).detach().numpy().transpose()
		q = q +h*dH[:dim]
		p = p +h*dH[dim:]
		return np.block([q,p])

def naiveTrajectoryNN(z,h,net,N=1):
	## trajectory computed with classicInt
  z = z.reshape(1,-1)[0]
  trj = np.zeros((len(z),N+1))
  trj[:,0] = z.copy()
  for j in tqdm(range(0,N)):
    trj[:,j+1] = naiveIntNN(trj[:,j].copy(),h,net)
  return trj[:, :-1], trj[:, 1:]
