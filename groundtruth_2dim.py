from tqdm import tqdm
import numpy as np
from skopt.space import Space
from skopt.sampler import Halton

def classicSympEuler(z,f1,f2,h,maxiters):
	## classical symplectic Euler scheme
	dim = int(len(z)/2)
	q=z[:dim]
	p=z[dim:]
	fstage = lambda stg: h * f1(np.block([q + stg, p]))

	stageold=np.zeros(dim) 
	stage = fstage(stageold) +0.
	Iter = 0

	while (np.amax(abs(stage - stageold)) > 1e-10 and Iter<int(maxiters)):
		stageold = stage+0.
		stage = fstage(stage)+0.
		Iter = Iter+1
	q = q+stage
	p = p + h*f2(np.block([q,p]))
	return np.block([q,p])

def SympEulerTrajectory(z,f1,f2,h,N=10,n_h=1,maxiters=100):
	## trajectory computed with classicInt
  h_gen = h/n_h
  z = z.reshape(1,-1)[0]
  trj = np.zeros((len(z),N+1))
  trj[:,0] = z.copy()

  for i in range(0,N):
    for j in range(0,int(n_h+1)):
      trj[:,i+1] = classicSympEuler(trj[:,i].copy(),f1,f2,h_gen,maxiters)
  return trj[:, :-1], trj[:, 1:]


def CreateTrainingDataTrajSympEuler(traj_len,ini_con,spacedim,h,f1,f2,seed,n_h = 800,maxiters=100):
  np.random.seed(seed = seed)
  startcon = np.random.uniform(spacedim[0][0], spacedim[0][1], size = ini_con)
  for i in range(len(spacedim)-1):
    startcon = np.vstack((startcon, np.random.uniform(spacedim[i+1][0], spacedim[i+1][1], size = ini_con)))
  h_gen = h/n_h
  finalcon = startcon.copy()
  if ini_con==1: return SympEulerTrajectory(startcon,f1,f2,h,traj_len,n_h,maxiters)
  else:
    start, final= SympEulerTrajectory(np.squeeze(startcon[:,0]),f1,f2,h,traj_len,n_h,maxiters)
    for k in range(ini_con-1):
      new_start, new_final = SympEulerTrajectory(np.squeeze(startcon[:,k+1]),f1,f2,h,traj_len,n_h,maxiters)
      start = np.hstack((start, new_start))
      final = np.hstack((final, new_final))
  return start,final

def classicLeapfrog(z,f1,f2,h):
## classical Leapfrog scheme for force field f
# can compute multiple initial values simultanously, z[k]=list of k-component of all initial values
	dim = int(len(z)/2)
	z[dim:] = z[dim:]+h/2*f2(z)
	z[:dim] = z[:dim]+h*f1(z)
	z[dim:] = z[dim:]+h/2*f2(z)
	return z

def LeapfrogTrajectory(z,f1,f2,h,N=10,n_h=100):
  ## trajectory computed with classicInt
  h_gen = h/n_h
  z = z.reshape(1,-1)[0]
  trj = np.zeros((len(z),N+1))
  trj[:,0] = z.copy()
  for i in range(0,N):
    for j in range(0,int(n_h+1)):
      trj[:,i+1] = classicLeapfrog(trj[:,i].copy(),f1,f2,h_gen)
  return trj[:, :-1], trj[:, 1:]

def CreateTrainingDataTrajLeapfrog(traj_len,ini_con,spacedim,h,f1,f2,seed,n_h = 800,maxiters=100):
  np.random.seed(seed = seed)
  startcon = np.random.uniform(spacedim[0][0], spacedim[0][1], size = ini_con)
  for i in range(len(spacedim)-1):
    startcon = np.vstack((startcon, np.random.uniform(spacedim[i+1][0], spacedim[i+1][1], size = ini_con)))
  h_gen = h/n_h
  finalcon = startcon.copy()
  if ini_con==1: return LeapfrogTrajectory(startcon,f1,f2,h,traj_len,n_h)
  else:
    start, final= LeapfrogTrajectory(np.squeeze(startcon[:,0]),f1,f2,h,traj_len,n_h)
    for k in range(ini_con-1):
      new_start, new_final = LeapfrogTrajectory(np.squeeze(startcon[:,k+1]),f1,f2,h,traj_len,n_h)
      start = np.hstack((start, new_start))
      final = np.hstack((final, new_final))
  return start,final

def get_within_array(trajectories, spacedim):
  within_array = np.asarray([])
  for i in range(len(trajectories)):
    np.sum(np.square(np.asarray([spacedim[0][0], spacedim[0][1]]), np.asarray([spacedim[1][0], spacedim[1][1]])))
    try:
      v = np.amin(np.concatenate((np.where(trajectories[i][1][0]<spacedim[0][0])[0],np.where(trajectories[i][1][0]>spacedim[0][1])[0], 
              np.where(trajectories[i][1][1]<spacedim[1][0])[0], np.where(trajectories[i][1][1]>spacedim[1][1])[0])))
      within_array = np.append(within_array, v)
    except ValueError:
      within_array = np.append(within_array, len(trajectories[i][1][1]))
  return within_array

# def CreateTrainingDataTrajClassicInt(traj_len,ini_con,spacedim,h,f1,f2,n_h = 800,maxiters=100):
#   space = Space(spacedim)
#   h_gen = h/n_h
#   halton = Halton()
#   startcon = np.array(halton.generate(space, ini_con)).transpose()
#   finalcon = startcon.copy()
#   # Compute flow map from Halton sequence to generate learning data
#   if ini_con==1: return classicTrajectory(startcon,f1,f2,h,traj_len,n_h,maxiters)
#   else:
#     start, final= classicTrajectory(np.squeeze(startcon[:,0]),f1,f2,h,traj_len,n_h,maxiters)
#     for k in range(ini_con-1):
#       new_start, new_final = classicTrajectory(np.squeeze(startcon[:,k+1]),f1,f2,h,traj_len,n_h,maxiters)
#       start = np.hstack((start, new_start))
#       final = np.hstack((final, new_final))
#   return start,final
