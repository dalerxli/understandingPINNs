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

def classicIntNN(z,h,net):
	## classical symplectic Euler scheme
		dim = int(len(z)/2)
		q=z[:dim]
		p=z[dim:]		
		fstage = lambda stg: h * torch.squeeze(net(torch.tensor([q+stg,p]).float().transpose(1,0)),0).detach().numpy().transpose()[:dim]

		stageold=np.zeros(dim) 
		stage = fstage(stageold) +0.
		Iter = 0

		while (np.amax(abs(stage - stageold)*25) > 1e-10 and Iter<100):
			stageold = stage+0.
			stage = fstage(stage)+0.
			Iter = Iter+1
		q = q+stage
		p = p +h*torch.squeeze(net(torch.tensor([q,p]).float().transpose(1,0)),0).detach().numpy().transpose()[dim:]
		return np.block([q,p])

def classicTrajectoryNN(z,h,net,N=1):
	## trajectory computed with classicInt
  z = z.reshape(1,-1)[0]
  trj = np.zeros((len(z),N+1))
  trj[:,0] = z.copy()
  for j in tqdm(range(0,N)):
    trj[:,j+1] = classicIntNN(trj[:,j].copy(),h,net)
  return trj[:, :-1], trj[:, 1:]
