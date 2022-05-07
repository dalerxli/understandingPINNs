import numpy as np
from numpy import sin, cos
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import time
import math

def MSE(arr, groundtruth, diagdist):
  assert arr.shape == groundtruth.shape, ("shape of array is", arr.shape, "shape of groundtruth is", groundtruth.shape)
  err = np.nan_to_num(((arr-groundtruth)**2), nan = diagdist)
  err[err>diagdist] = diagdist
  return np.mean(err)


def get_grad(model, z,device):
		inputs=Variable(torch.tensor([z[0][0],z[1][0]]), requires_grad = True).to(device)
		out=model(inputs.float())
		dH=torch.autograd.grad(out, inputs, grad_outputs=out.data.new(out.shape).fill_(1),create_graph=True)[0]
		return dH.detach().cpu().numpy()[1], -dH.detach().cpu().numpy()[0] # negative dH/dq is dp/dt

def classicIntNNH_autograd(z,h,model,device):
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

def classicTrajectoryNNH_autograd(z,h,model,device,N=1):
	## trajectory computed with classicInt
  z = z.reshape(1,-1)[0]
  trj = np.zeros((len(z),N+1))
  trj[:,0] = z.copy()
  for j in range(0,N):
    trj[:,j+1] = classicIntNNH_autograd(trj[:,j].reshape(-1,1).copy(),h,model,device)
  return trj[:, :-1], trj[:, 1:]

def compute_metrics_PINN(nn, device, h, diagdist, xshort, yshort, xlong, ylong, eval_len, len_within, len_short):
    # def compute_metrics_PINN(nn, device, h, diagdist, xshort, yshort, xlong, ylong, eval_len, len_within, long_groundtruth, len_short, truevector):
    # results_start = np.asarray(classicTrajectoryNNH_autograd(np.asarray([[0.4],[0.]]),h,model=nn,device=device,N=eval_len)) 
    # withinspace_longtraj_symplectic_MSe = MSE(long_groundtruth[0,1,:,:], results_start[1,:,:], diagdist)
    # results_start = np.asarray(naiveTrajectoryNNH_autograd(np.asarray([[0.4],[0.]]),h,model=nn,device=device,N=eval_len))
    # withinspace_longtraj_naive_MSe = MSE(long_groundtruth[0,1,:,:], results_start[1,:,:], diagdist)

    MSE_long, time_long, MSE_within, time_within, MSE_onestep, time_onestep = 0.,0.,0.,0.,0.,0.
    count = 1
    for i in tqdm(np.expand_dims(np.c_[np.ravel(xlong),np.ravel(ylong)],2)):
      groundtruth_H = H(i)
      starttime = time.time()
      results_start = np.asarray(classicTrajectoryNNH_autograd(i,h,model=nn,device=device,N=eval_len)) 
      time_long += time.time()-starttime
      MSE_long += np.mean(H(results_start[0])-groundtruth_H)
      steps = int(len_within[count-1])
      supp = (len_within>0).sum()
      if steps == 0:
        pass
      else: 
        starttime = time.time()
        results_start = np.asarray(classicTrajectoryNNH_autograd(i,h,model=nn,device=device,N=steps)) 
        time_within += time.time()-starttime
        MSE_within += np.mean(H(results_start[0])-groundtruth_H)
      count+=1 
    count = 1
    for i in tqdm(np.expand_dims(np.c_[np.ravel(xshort),np.ravel(yshort)],2)):
      groundtruth_H = H(i)
      starttime = time.time()
      results_start = np.asarray(classicTrajectoryNNH_autograd(i,h,model=nn,device=device,N=1)) 
      time_onestep += time.time()-starttime
      MSE_onestep += np.mean(H(results_start[0])-groundtruth_H)
      count+=1
    return MSE_long/25, time_long, MSE_within/supp, time_within, MSE_onestep/400, time_onestep
