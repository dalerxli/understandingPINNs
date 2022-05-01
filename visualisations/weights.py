import seaborn as sns 
import matplotlib.cm
import numpy as np
import matplotlib.pyplot as plt



def heatmap_creation(params,dir,filename):
  x,y=0,0
  flag = 1
  arr = np.zeros((x,y))
  show_annot_array = np.full((x, y), False)
  coords = [0,0]
  for name, param in params:
    if param.requires_grad and "weight" in name:
        if x==0 and y==0: 
          x,y = x+param.data.shape[0],y+param.data.shape[1]
          new_arr = param.data.detach().numpy()
          new_annot_array = np.full((x, y), True)
          coords[1] += param.data.shape[1]
        elif flag == 1: 
          y=y+param.data.shape[1]
          flag*=-1
          new_arr = np.zeros((x,y))
          new_annot_array = np.full((x, y), False)
          new_arr[:,:y-param.data.shape[1]] = arr 
          new_annot_array[:,:y-param.data.shape[1]] = show_annot_array
          new_arr[x-param.data.shape[0]:x,y-param.data.shape[1]:y] = param.data.detach().numpy() 
          new_annot_array[x-param.data.shape[0]:x,y-param.data.shape[1]:y] = True
        else:
          x=x+param.data.shape[0]
          flag*=-1
          new_arr = np.zeros((x,y))
          new_annot_array = np.full((x, y), False)
          new_arr[:x-param.data.shape[0],:] = arr 
          new_annot_array[:x-param.data.shape[0],:] = show_annot_array
          new_arr[x-param.data.shape[0]:x,y-param.data.shape[1]:y] = param.data.detach().numpy() 
          new_annot_array[x-param.data.shape[0]:x,y-param.data.shape[1]:y] = True
        arr = new_arr
        show_annot_array = new_annot_array
  fig, ax = plt.subplots(figsize = (20,10))
  sns.heatmap(data = arr, ax = ax, cmap = cm.bwr, vmin=-1., vmax=1., center = 0., annot=True)
  for text, show_annot in zip(ax.texts, (element for row in show_annot_array for element in row)):
    text.set_visible(show_annot)
  plt.savefig(dir+filename+".png")
  return 

# class TestNet(nn.Module):

#     def __init__(self, input_size, hidden_size, output_size):
#         super(TestNet , self).__init__()
#         self.hidden_layer_1 = nn.Linear( input_size, hidden_size, bias=True)
#         self.hidden_layer_2 = nn.Linear( hidden_size, hidden_size, bias=True)
#         self.hidden_layer_3 = nn.Linear( hidden_size, hidden_size, bias=True)
#         self.hidden_layer_4 = nn.Linear( hidden_size, hidden_size, bias=True)
#         self.hidden_layer_5 = nn.Linear( hidden_size, hidden_size, bias=True)
#         self.output_layer = nn.Linear( hidden_size, output_size , bias=True)
        
#     def forward(self, x):
#         x = softplus(self.hidden_layer_1(x)) # F.relu(self.hidden_layer_1(x)) # 
#         x = softplus(self.hidden_layer_2(x)) # F.relu(self.hidden_layer_2(x)) # 
#         x = softplus(self.hidden_layer_3(x)) # F.relu(self.hidden_layer_2(x)) # 
#         x = softplus(self.hidden_layer_4(x)) # F.relu(self.hidden_layer_2(x)) # 
#         x = softplus(self.hidden_layer_5(x)) # F.relu(self.hidden_layer_2(x)) # 
#         x = self.output_layer(x)

#         return x
      
# testnet=TestNet(2,4,1)

# print(heatmap_creation(testnet.named_parameters()))
