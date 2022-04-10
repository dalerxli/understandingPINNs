import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import os
import time

from sklearn.model_selection import train_test_split
def data_preprocessing(start_train, final_train,device):       
    # wholemat=[]
    # for i in range(len(start_train[0,:])):
    #     wholemat.append(np.vstack((
    #         np.hstack((start_train[:,i], (final_train[:,i]-start_train[:,i])/h)),
    #         np.hstack((final_train[:,i], (final_train[:,i]-start_train[:,i])/h)))))
    wholemat = np.hstack((start_train.transpose(), final_train.transpose()))

    wholemat =torch.tensor(wholemat)
    wholemat=wholemat.to(device)

    wholemat,evalmat=train_test_split(wholemat, train_size=0.8, random_state=1)

    return wholemat,evalmat

# define model
def softplus(x):
    return torch.log(torch.exp(x)+1)

from sklearn.model_selection import train_test_split


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Net , self).__init__()
        self.hidden_layer_1 = nn.Linear( input_size, hidden_size, bias=True)
        self.hidden_layer_2 = nn.Linear( hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear( hidden_size, output_size , bias=True)
        
    def forward(self, x):
        x = softplus(self.hidden_layer_1(x)) # F.relu(self.hidden_layer_1(x)) # 
        x = softplus(self.hidden_layer_2(x)) # F.relu(self.hidden_layer_2(x)) # 
        x = self.output_layer(x)

        return x

class sepNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(sepNet , self).__init__()
        self.net1 = Net(int(input_size/2), hidden_size, output_size)  
        self.net2 = Net(int(input_size/2), hidden_size, output_size) 
        
    def forward(self, x):
        x1,x2 = torch.unsqueeze(x[:,0],1),torch.unsqueeze(x[:,1],1)
        x1 = self.net1(x1) 
        x2 = self.net2(x2) 
        x = torch.add(x1, x2)

        return x

def lossfuc(model,mat,x,y,device,x0,H0,dim,c1=1,c2=1,c3=1,c4=1,verbose=False):
    f3=(model(torch.tensor([[x0,x0]]).to(device))-torch.tensor([[H0]]).to(device))**2
    dH=torch.autograd.grad(y, x, grad_outputs=y.data.new(y.shape).fill_(1),create_graph=True)[0]
    dHdq=dH[:,0]
    dHdp=dH[:,1]
    qprime=(mat[:,2])
    pprime=(mat[:,3])
    f1=torch.mean((dHdp-qprime)**2,dim=0)
    f2=torch.mean((dHdq+pprime)**2,dim=0)
    f4=torch.mean((dHdq*qprime+dHdp*pprime)**2,dim=0)
    loss=torch.mean(c1*f1+c2*f2+c3*f3+c4*f4)
    meanf1,meanf2,meanf3,meanf4=torch.mean(c1*f1),torch.mean(c2*f2),torch.mean(c3*f3),torch.mean(c4*f4)
    if verbose:
      print(x)
      print(meanf1,meanf2,meanf3,meanf4)
      print(loss,meanf1,meanf2,meanf3,meanf4)
    return loss,meanf1,meanf2,meanf3,meanf4

def data_preprocessing(start_train, final_train,device):       
    # wholemat=[]
    # for i in range(len(start_train[0,:])):
    #     wholemat.append(np.vstack((
    #         np.hstack((start_train[:,i], (final_train[:,i]-start_train[:,i])/h)),
    #         np.hstack((final_train[:,i], (final_train[:,i]-start_train[:,i])/h)))))
    wholemat = np.hstack((start_train.transpose(), final_train.transpose()))

    wholemat =torch.tensor(wholemat)
    wholemat=wholemat.to(device)

    wholemat,evalmat=train_test_split(wholemat, train_size=0.8, random_state=1)

    return wholemat,evalmat

## train

# evaluate loss of dataset use c1,c2,c3,c4=1,10,1,1
def get_loss(model,device,initial_conditions,bs,x0,H0,dim,wholemat,evalmat,trainset=False,verbose=False):
    # this function is used to calculate average loss of a whole dataset
    # rootpath: path of set to be calculated loss
    # model: model
    # trainset: is training set or not


    if trainset:
        mat=wholemat
    else:
        mat=evalmat
    avg_loss=0
    avg_f1=0
    avg_f2=0
    avg_f3=0
    avg_f4=0
    for count in range(0,len(mat),bs):
      curmat=mat[count:count+bs]
      x=Variable((curmat[:,:dim]).float(),requires_grad=True)
      y=model(x)
      x=x.to(device)
      loss,f1,f2,f3,f4=lossfuc(model,curmat,x,y,device,x0,H0,dim)
      avg_loss+=loss.detach().cpu().item()
      avg_f1+=f1.detach().cpu().item()
      avg_f2+=f2.detach().cpu().item()
      avg_f3+=f3.detach().cpu().item()
      avg_f4+=f4.detach().cpu().item()
    num_batches=len(mat)//bs
    avg_loss/=num_batches
    avg_f1/=num_batches
    avg_f2/=num_batches
    avg_f3/=num_batches
    avg_f4/=num_batches
    if verbose:
        print(' loss=',avg_loss,' f1=',avg_f1,' f2=',avg_f2,' f3=',avg_f3,' f4=',avg_f4)
    return avg_loss


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if abs(self.counter-self.patience)<5:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')     # 这里会存储迄今最优模型的参数
        torch.save(model, 'checkpoint.pt')                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss

def train(net,bs,num_epoch,initial_conditions,device,wholemat,evalmat,x0,H0,dim,LR,patience,c1,c2,c3,c4):
    # function of training process
    # net: the model
    # bs: batch size 
    # num_epoch: max of epoch to run
    # initial_conditions: number of trajectory in train set
    # patience: EarlyStopping parameter
    # c1~c4: hyperparameter for loss function


    avg_lossli,avg_f1li,avg_f2li,avg_f3li,avg_f4li=[],[],[],[],[]
    avg_vallosses=[]
    
    start = time.time()
    lr = LR # initial learning rate
    net=net.to(device)

    early_stopping = EarlyStopping(patience=patience, verbose=False,delta=0.00001) # delta
    optimizer=torch.optim.Adam( net.parameters() , lr=lr )
    for epoch in range(num_epoch):

        running_loss=0

        running_f1=0
        running_f2=0
        running_f3=0
        running_f4=0
        num_batches=0
        
        # train
        shuffled_indices=torch.randperm(len(wholemat))
        net.train()
        for count in range(0,len(wholemat),bs):
            optimizer.zero_grad()

            indices=shuffled_indices[count:count+bs]
            mat=wholemat[indices]

            x=Variable(torch.tensor(mat[:,:dim]).float(),requires_grad=True)
            y=net(x)

            loss,f1,f2,f3,f4=lossfuc(net,mat,x,y,device,x0,H0,dim,c1,c2,c3,c4)  
            loss.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), 1)

            optimizer.step()

            # compute some stats
            running_loss += loss.detach().item()
            running_f1 += f1.detach().item()
            running_f2 += f2.detach().item()
            running_f3 += f3.detach().item()
            running_f4 += f4.detach().item()

            num_batches+=1
            torch.cuda.empty_cache()



        avg_loss = running_loss/num_batches
        avg_f1 = running_f1/num_batches
        avg_f2 = running_f2/num_batches
        avg_f3 = running_f3/num_batches
        avg_f4 = running_f4/num_batches
        elapsed_time = time.time() - start
        
        avg_lossli.append(avg_loss)
        avg_f1li.append(avg_f1)
        avg_f2li.append(avg_f2)
        avg_f3li.append(avg_f3)
        avg_f4li.append(avg_f4)
        
        
        # evaluate
        net.eval()
        avg_val_loss=get_loss(net,device,len(evalmat),bs,x0,H0,dim,wholemat,evalmat)
        avg_vallosses.append(avg_val_loss)
        
        if epoch % 10 == 0 : 
            print(' ')
            print('epoch=',epoch, ' time=', elapsed_time,
                  ' loss=', avg_loss ,' val_loss=',avg_val_loss,' f1=', avg_f1 ,' f2=', avg_f2 ,
                  ' f3=', avg_f3 ,' f4=', avg_f4 ,'percent lr=', optimizer.param_groups[0]["lr"] )
        
        
        
        early_stopping(avg_val_loss,net)
        if early_stopping.early_stop:
            print('Early Stopping')
            break
            
    net=torch.load('checkpoint.pt')
    return net,avg_vallosses,avg_lossli,avg_f1li,avg_f2li,avg_f3li,avg_f4li
  
def get_grad(model, z,device):
		inputs=Variable(torch.tensor(z), requires_grad = True).to(device)
		out=model(torch.transpose(inputs,1,0).float())
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

def LeapfrogNNH_autograd(z,h,model,device):
## classical Leapfrog scheme for force field f
# can compute multiple initial values simultanously, z[k]=list of k-component of all initial values
	dim = int(len(z)/2)
	z[dim:] = z[dim:]+h/2*get_grad(model, z,device)[1]
	z[:dim] = z[:dim]+h*z[dim:]
	z[dim:] = z[dim:]+h/2*get_grad(model, z,device)[1]
	return z
  
def gen_one_trajNNH_autograd(traj_len,start,h,model,device,n_h = 800):
  h_gen = h/n_h
  x, final = start.copy(), start.copy()
  for i in range(traj_len):
    start=np.hstack((start,x))
    for j in range(0,int(n_h)):
      x=LeapfrogNNH_autograd(x,h_gen,model,device)
    final=np.hstack((final,x))
  return start[:,1:],final[:,1:]

def compute_metrics_sepNN(nn, device, h, diagdist, xshort, yshort, xlong, ylong, eval_len, len_within, long_groundtruth, len_short, truevector):
    results_start = np.asarray(classicTrajectoryNNH_autograd(np.asarray([[0.4],[0.]]),h,model=nn,device=device,N=eval_len)) 
    withinspace_longtraj_symplectic_MSe = MSE(long_groundtruth[0,1,:,:], results_start[1,:,:], diagdist)
    results_start = np.asarray(naiveTrajectoryNNH_autograd(np.asarray([[0.4],[0.]]),h,model=nn,device=device,N=eval_len))
    withinspace_longtraj_naive_MSe = MSE(long_groundtruth[0,1,:,:], results_start[1,:,:], diagdist)

    MSE_long, time_long, MSE_long_naive, time_long_naive, MSE_within, time_within, MSE_within_naive, time_within_naive, MSE_onestep, time_onestep, MSE_vectorfield, time_vectorfield, MSE_long_leapfrog, time_long_leapfrog, MSE_within_leapfrog, time_within_leapfrog, MSE_onestep_leapfrog, time_onestep_leapfrog, MSE_onestep_naive, time_onestep_naive = 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.
    count = 1
    for i in tqdm(np.expand_dims(np.c_[np.ravel(xlong),np.ravel(ylong)],2)):
      starttime = time.time()
      results_start = np.asarray(classicTrajectoryNNH_autograd(i,h,model=nn,device=device,N=eval_len)) 
      time_long += time.time()-starttime
      MSE_long += MSE(long_groundtruth[count,1,:,:], results_start[1,:,:], diagdist)
      starttime = time.time()
      results_start = np.asarray(naiveTrajectoryNNH_autograd(i,h,model=nn,device=device,N=eval_len,))
      time_long_naive += time.time()-starttime
      MSE_long_naive += MSE(long_groundtruth[count,1,:,:], results_start[1,:,:], diagdist)
      starttime = time.time()
      results_start = np.asarray(gen_one_trajNNH_autograd(eval_len,i,h,model=nn,device=device,))
      time_long_leapfrog += time.time()-starttime
      MSE_long_leapfrog += MSE(long_groundtruth[count,1,:,:], results_start[1,:,:], diagdist)
      steps = int(len_within[count-1])
      supp = (len_within>0).sum()
      if steps == 0:
        pass
      else: 
        starttime = time.time()
        results_start = np.asarray(classicTrajectoryNNH_autograd(i,h,model=nn,device=device,N=steps)) 
        time_within += time.time()-starttime
        MSE_within += MSE(long_groundtruth[count,1,:,:steps], results_start[1,:,:], diagdist)
        starttime = time.time()
        results_start = np.asarray(naiveTrajectoryNNH_autograd(i,h,model=nn,device=device,N=steps,))
        time_within_naive += time.time()-starttime
        MSE_within_naive += MSE(long_groundtruth[count,1,:,:steps], results_start[1,:,:], diagdist)
	starttime = time.time()
        results_start = np.asarray(gen_one_trajNNH_autograd(eval_len,i,h,model=nn,device=device)) 
        time_within_leapfrog += time.time()-starttime
        MSE_within_leapfrog += MSE(long_groundtruth[count,1,:,:steps], results_start[1,:,:], diagdist)
      count+=1 
    count = 1
    for i in tqdm(np.expand_dims(np.c_[np.ravel(xshort),np.ravel(yshort)],2)):
      starttime = time.time()
      results_start = np.asarray(classicTrajectoryNNH_autograd(i,h,model=nn,device=device,N=1)) 
      time_onestep += time.time()-starttime
      MSE_onestep += MSE(len_short[count,1,:,:], results_start[1,:,:], diagdist)
      starttime = time.time()
      results_start = np.asarray(gen_one_trajNNH_autograd(1, i,h,model=nn,device=device)) 
      time_onestep_leapfrog += time.time()-starttime
      MSE_onestep_leapfrog += MSE(len_short[count,1,:,:], results_start[1,:,:], diagdist)    
      starttime = time.time()
      results_start = np.asarray(naiveTrajectoryNNH_autograd(i,h,model=nn,device=device,N=1,))
      time_onestep_naive += time.time()-starttime
      MSE_onestep_naive += MSE(len_short[count,1,:,:], results_start[1,:,:], diagdist)  
      starttime = time.time()
      dH = get_grad(nn, i,device)
      vectorfield = np.asarray([dH[0],dH[1]])
      time_vectorfield += time.time()-starttime
      MSE_vectorfield += MSE(truevector(len_short[count,0,:,:].flatten()), vectorfield, diagdist)
      count+=1
    return MSE_long/25, time_long, MSE_long_naive/25, time_long_naive, MSE_within/supp, time_within, MSE_within_naive/supp, time_within_naive, MSE_onestep/400, time_onestep, MSE_vectorfield/400, time_vectorfield/400, withinspace_longtraj_symplectic_MSe, withinspace_longtraj_naive_MSe, MSE_long_leapfrog/25, time_long_leapfrog, MSE_within_leapfrog/supp, time_within_leapfrog, MSE_onestep_leapfrog/400, time_onestep_leapfrog, MSE_onestep_naive/400, time_onestep_naive, 
