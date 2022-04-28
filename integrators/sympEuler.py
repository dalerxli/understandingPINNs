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

def classicIntNN(z,h,net):
	## classical symplectic Euler scheme
	dim = int(len(z)/2)
	q=z[:dim]
	p=z[dim:]		
	fstage = lambda stg: h * get_grad(model, np.concatenate([q+stg,p]),device)[0]

	stageold=np.zeros(dim) 
	stage = fstage(stageold) +0.
	Iter = 0

	while (np.amax(abs(stage - stageold)) > 1e-8 and Iter<100):
		stageold = stage+0.
		stage = fstage(stage)+0.
		Iter = Iter+1
	q = q+stage
	p = p + h*get_grad(model, np.concatenate([q,p]),device)[1]
	return np.block([q,p])

def classicTrajectoryNN(z,h,net,N=1):
	## trajectory computed with classicInt
	z = z.reshape(1,-1)[0]
	trj = np.zeros((len(z),N+1))
	trj[:,0] = z.copy()
	for j in range(0,N):
		trj[:,j+1] = classicIntNN(trj[:,j].reshape(-1,1).copy(),h,model,device)
	return trj[:, :-1], trj[:, 1:]
