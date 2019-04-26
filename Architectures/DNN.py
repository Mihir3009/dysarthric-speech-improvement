
import numpy as np
from os import listdir, makedirs, getcwd, remove, sys
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms, datasets, models
from torch import Tensor

import visdom
import math
import matplotlib.pyplot as plt 
import scipy
import h5py
from scipy import io as sio
from scipy.io import savemat
from scipy.io import loadmat

#import encoder as s

viz = visdom.Visdom()

#parameters

batch_size=1

# print("\n\n\n\n\nCuda available:",torch.cuda.is_available(),"\n\n\n\n\n")

f_arg= sys.argv[1]
s_arg= sys.argv[2]


class dnn(nn.Module):
    
    # Weight Initialization [we initialize weights here]
    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.out.weight)

        nn.init.xavier_uniform_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.bias)
        nn.init.xavier_uniform_(self.out.bias)
    
    # Layers 
    def __init__(self, G_in, G_out, w1, w2, w3, w4):
        super(dnn, self).__init__()
        
        self.fc1= nn.Linear(G_in, w1)
        self.fc2= nn.Linear(w1, w2)
        self.fc3= nn.Linear(w2, w3)
        self.fc4= nn.Linear(w3, w4)
        self.out= nn.Linear(w4, G_out)

        # self.weight_init()
    
    # Deep neural network [you are passing data layer-to-layer]
    def forward(self, x):
        
        # x = x.view(x.size(0), -1)   #[If you want to reshape]
        #x.type(torch.cuda.FloatTensor)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.out(x)
        # x = x.view(1, 1, 1000, 25)   #[If you want to reshape]
        return x


class encoder(nn.Module):

    def __init__(self, G_in, G_out, w1, w2, w3):
        super(encoder, self).__init__()
        
        self.fc1 = nn.Linear(G_in, w1)
        self.fc2 = nn.Linear(w1, w2)
        self.fc3 = nn.Linear(w2, w3)
        self.out = nn.Linear(w3, G_out)

        
    def forward(self, x):       
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x

class decoder(nn.Module):

    def __init__(self, G_in, G_out, w1, w2, w3):
        super(decoder, self).__init__()
        
        self.fc1= nn.Linear(G_in, w1)
        self.fc2= nn.Linear(w1, w2)
        self.fc3= nn.Linear(w2, w3)
        self.out= nn.Linear(w3, G_out)

        
    def forward(self, x):       
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x    


def feature_extraction(dys):

    load_en = torch.load(join(mainfolder1, "en_Ep_50.pth"), map_location='cpu')
    load_de = torch.load(join(mainfolder1, "de_Ep_50.pth"), map_location='cpu')

    out_en = load_en(dys)
    out_de = load_de(out_en)

    return out_de

class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x




# Class for load the data into system
class speech_data(Dataset):
    
    def __init__(self, folder_path):
        self.path = folder_path
        self.files = listdir(folder_path)
        self.length = len(self.files)
        
    def __getitem__(self, index):
        d = loadmat(join(self.path, self.files[int(index)]))
        return np.array(d['Feat']), np.array(d['Clean_cent'])
    
    def __len__(self):
        return self.length


        
# Path where you want to store your results
mainfolder = "/media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/result/"+f_arg+"/"+s_arg+"/model/"
mainfolder1 = "/media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/result/encoder/"+s_arg+"/model/"
# Training Data path
traindata = speech_data(folder_path="/media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/batches/"+s_arg+"/training_batches")
train_dataloader = DataLoader(dataset=traindata, batch_size=1, shuffle=True, num_workers=2)

# Path for validation data
valdata = speech_data(folder_path="/media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/batches/"+s_arg+"/validation_batches")
val_dataloader = DataLoader(dataset=valdata, batch_size=1, shuffle=True, num_workers=2)

# Loss function
mmse_loss = nn.MSELoss()

# initialize the Deep neural network
net = dnn(40, 40, 1024, 1024, 512, 512)

# Optimizer [Adam]
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)


# Function for training the data

def training(data_loader, n_epochs):
    net.train()
    
    for en, (a, b) in enumerate(data_loader):
        a = Variable(a.squeeze(0)).type(torch.FloatTensor)
        b = Variable(b.squeeze(0)).type(torch.FloatTensor)

        enh_dys = feature_extraction(a)

        optimizer.zero_grad()
        out = net(enh_dys)
        loss = mmse_loss(out, b)*5
        
        loss.backward()
        optimizer.step()

        
        print ("[Epoch: %d] [Iter: %d/%d] [Loss: %f]" % (n_epochs, en, len(data_loader), loss.cpu().data.numpy()))
    

# Function that validate our model after every epoch 

def validating(data_loader):
    net.eval()
    running_loss = 0
    for en, (a, b) in enumerate(data_loader):
        a = Variable(a.squeeze(0)).type(torch.FloatTensor)
        b = Variable(b.squeeze(0)).type(torch.FloatTensor)

        enh_dys = feature_extraction(a)

        out = net(enh_dys)  
        loss = mmse_loss(out, b)*5

        running_loss += loss.item()
        
    return running_loss/(en+1)
    
    
 
# For traning, it is 'True'. For testing, make it 'False'

isTrain = False


if isTrain:
    epoch = 100
    arr = []
    for ep in range(epoch):

        training(train_dataloader, ep+1)

        if (ep+1)%5==0:
            torch.save(net, join(mainfolder,"net_Ep_{}.pth".format(ep+1)))

        gl = validating(val_dataloader)
        print("loss: " + str(gl))
        arr.append(gl)
        if ep == 0:
            gplot = viz.line(Y=np.array([gl]), X=np.array([ep]), opts=dict(title='dnn_MCEP'))
        else:
            viz.line(Y=np.array([gl]), X=np.array([ep]), win=gplot, update='append')

            
    savemat(mainfolder+"/"+str('dnn.mat'),  mdict={'foo': arr})

    plt.figure(1)
    plt.plot(arr)
    plt.savefig(mainfolder+'/dnn.png')




# Tesing Time this code will run
else:
    print("Testing")
    
    # Path where you want to store data
    save_folder = "/media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/result/"+f_arg+"/"+s_arg+"/mask/"
    
    # Path for testing data
    test_folder_path="/media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/batches/"+s_arg+"/dysarthric/testing_batches/"
    
    n = len(listdir(test_folder_path))
    Gnet = torch.load(join(mainfolder,"net_Ep_100.pth"), map_location='cpu')
    for i in range(n):
        d = loadmat(join(test_folder_path, "Batch_{}.mat".format(str(i))))
        a = torch.from_numpy(d['Feat'])
        a = Variable(a.squeeze(0).type('torch.FloatTensor'))
        enh_dys = feature_extraction(a)
        Gout = Gnet(enh_dys)
        # np.save(join(save_folder,'Test_Batch_{}.npy'.format(str(i))), Gout.cpu().data.numpy())
        savemat(join(save_folder,'Test_Batch_{}.mat'.format(str(i))),  mdict={'foo': Gout.cpu().data.numpy()})





