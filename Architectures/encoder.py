
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


f_arg= sys.argv[1]
s_arg= sys.argv[2]



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



mainfolder = "/media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/result/"+f_arg+"/"+s_arg+"/model/"
train_data = speech_data(folder_path="/media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/batches/"+s_arg+"/training_batches")
train_dataloader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=2)


en = encoder(40, 128, 256, 512, 512)
de = decoder(128, 40, 256, 256, 128)

mse_loss = nn.MSELoss()

optimizer_en = torch.optim.Adam(en.parameters(), lr=0.0001)
optimizer_de = torch.optim.Adam(de.parameters(), lr=0.0001)



def training(data_loader, epoch):
    en.train()
    de.train()
    print("Here")
    for ep, (a,b) in enumerate(data_loader):
        a = Variable(a.squeeze(0)).type(torch.FloatTensor)
        b = Variable(b.squeeze(0)).type(torch.FloatTensor)

        optimizer_en.zero_grad()
        optimizer_de.zero_grad()

        en_out = en(b)
        de_out = de(en_out)
        loss = mse_loss(de_out, b)*10

        loss.backward()
        optimizer_en.step()
        optimizer_de.step()

        print ("[Epoch: %d] [Iter: %d/%d] [Loss: %f]" % (epoch, ep, len(data_loader), loss.cpu().data.numpy()))



def feature_extraction(dys):
    print("test")
    load_en = torch.load(join(mainfolder, "en_Ep_50.pth"), map_location='cpu')
    load_de = torch.load(join(mainfolder, "de_Ep_50.pth"), map_location='cpu')

    out_en = load_en(dys)
    out_de = load_de(out_en)

    return out_de



epoch = 50
for ep in range(epoch):
    training(train_dataloader, ep+1)
    if (ep+1)%5==0:
        torch.save(en, join(mainfolder,"en_Ep_{}.pth".format(ep+1)))
        torch.save(de, join(mainfolder,"de_Ep_{}.pth".format(ep+1)))
