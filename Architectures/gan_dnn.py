
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
#from encoder import *

viz = visdom.Visdom()
#parameters

batch_size=1

# print("\n\n\n\n\nCuda available:",torch.cuda.is_available(),"\n\n\n\n\n")


f_arg= sys.argv[1]
s_arg= sys.argv[2]


class generator(nn.Module):
    
    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.out.weight)

        nn.init.xavier_uniform_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.bias)
        nn.init.xavier_uniform_(self.fc4.bias)
        nn.init.xavier_uniform_(self.out.bias)

    def __init__(self, G_in, G_out, w1, w2, w3, w4):
        super(generator, self).__init__()
        
        self.fc1= nn.Linear(G_in, w1)
        self.fc2= nn.Linear(w1, w2)
        self.fc3= nn.Linear(w2, w3)
        self.fc4= nn.Linear(w3, w4) 
        self.out= nn.Linear(w4, G_out)

        # self.weight_init()
        
    def forward(self, x):
        
        # x = x.view(x.size(0), -1)
        #x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.out(x)
        # x = x.view(1, 1, 1000, 25)
        return x
        

class discriminator(nn.Module):

    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.out.weight)
    
    def __init__(self, D_in, D_out, w1, w2, w3, w4):
        super(discriminator, self).__init__()
        
        self.fc1= nn.Linear(D_in, w1)
        self.fc2= nn.Linear(w1, w2)
        self.fc3= nn.Linear(w2, w3)
        self.fc4= nn.Linear(w3, w4)
        self.out= nn.Linear(w4, D_out)

        # self.weight_init()
        
    def forward(self, y):
        
        # y = y.view(y.size(0), -1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = F.relu(self.fc4(y))
        y = F.sigmoid(self.out(y))
        return y

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


class test_speech_data(Dataset):
    
    def __init__(self, folder_path):
        self.path = folder_path
        self.files = listdir(folder_path)
        self.length = len(self.files)
        
    def __getitem__(self, index):
        # result = np.zeros((1000,275))
        d = loadmat(join(self.path, self.files[int(index)]))
        print(index)
        # result[:d.shape[0],:d.shape[1]] = d 
        return np.array(d['Feat'])[:,125:150]
    
    def __len__(self):
        return self.length
        



mainfolder = "/media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/result/"+f_arg+"/"+s_arg+"/model"
mainfolder1 = "/media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/result/encoder/"+s_arg+"/model/"

traindata = speech_data(folder_path="/media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/batches/"+s_arg+"/training_batches")
train_dataloader = DataLoader(dataset=traindata, batch_size=1, shuffle=True, num_workers=2)

valdata = speech_data(folder_path="/media/mihir/Dysarthia/dysarthic_interspeech/UA/gender_specific/batches/"+s_arg+"/validation_batches")
val_dataloader = DataLoader(dataset=valdata, batch_size=1, shuffle=True, num_workers=2)

adversarial_loss = nn.BCELoss()
mmse_loss = nn.MSELoss()

Gnet = generator(40, 40, 1024, 1024, 512, 512)
Dnet = discriminator(40, 1, 1024, 1024, 512, 512)

# Optimizers
optimizer_G = torch.optim.Adam(Gnet.parameters(), lr=0.0001)
optimizer_D = torch.optim.Adam(Dnet.parameters(), lr=0.0001)




def training(data_loader, n_epochs):
    Gnet.train()
    Dnet.train()
    
    for en, (a, b) in enumerate(data_loader):
        
        a = Variable(a.squeeze(0)).type(torch.FloatTensor)
        b = Variable(b.squeeze(0)).type(torch.FloatTensor)
        #a = Variable(a.squeeze(0))
        #print(b.size())
        #b = Variable(b.squeeze(0))
        #print(b.size())

        valid = Variable(Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False)

        optimizer_G.zero_grad()
        enh_dys = feature_extraction(a)
        Gout = Gnet(enh_dys)
        G_loss = adversarial_loss(Dnet(Gout), valid)*5
        
        G_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        real_loss = adversarial_loss(Dnet(b), valid)
        fake_loss = adversarial_loss(Dnet(Gout.detach()), fake)
        D_loss = (real_loss + fake_loss) / 2

        D_loss.backward()
        optimizer_D.step()
        
        print ("[Epoch: %d] [Iter: %d/%d] [D loss: %f] [G loss: %f]" % (n_epochs, en, len(data_loader), D_loss.cpu().data.numpy(), G_loss.cpu().data.numpy()))
    


 
def validating(data_loader):
    Gnet.eval()
    Dnet.eval()

    Grunning_loss = 0
    Drunning_loss = 0

    valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)
    
    for en, (a, b) in enumerate(data_loader):

        a = Variable(a.squeeze(0)).type(torch.FloatTensor)
        b = Variable(b.squeeze(0)).type(torch.FloatTensor)
        #a = Variable(a.squeeze(0))
        #b = Variable(b.squeeze(0))

        valid = Variable(Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False)
        
        enh_dys = feature_extraction(a)
        Gout = Gnet(enh_dys)
        G_loss = adversarial_loss(Dnet(Gout), valid)*5
        
        Grunning_loss += G_loss.item()
        
        real_loss = adversarial_loss(Dnet(b), valid)
        fake_loss = adversarial_loss(Dnet(Gout.detach()), fake)
        D_loss = (real_loss + fake_loss) / 2

        Drunning_loss += D_loss.item()
        
    return Drunning_loss/(en+1),Grunning_loss/(en+1)
    
    



isTrain = False


if isTrain:
    epoch = 100
    dl_arr = []
    gl_arr = []
    for ep in range(epoch):

        training(train_dataloader, ep+1)
        if (ep+1)%5==0:
            torch.save(Gnet, join(mainfolder,"gen_g_{}_d_{}_Ep_{}.pth".format(5,1,ep+1)))
            torch.save(Dnet, join(mainfolder,"dis_g_{}_d_{}_Ep_{}.pth".format(5,1,ep+1)))
        dl,gl = validating(val_dataloader)
        print("D_loss: " + str(dl) + " G_loss: " + str(gl))
        dl_arr.append(dl)
        gl_arr.append(gl)
        if ep == 0:
            gplot = viz.line(Y=np.array([gl]), X=np.array([ep]), opts=dict(title='Generator'))
            dplot = viz.line(Y=np.array([dl]), X=np.array([ep]), opts=dict(title='Discriminator'))
        else:
            viz.line(Y=np.array([gl]), X=np.array([ep]), win=gplot, update='append')
            viz.line(Y=np.array([dl]), X=np.array([ep]), win=dplot, update='append')

            
    savemat(mainfolder+"/"+str('discriminator_loss.mat'),  mdict={'foo': dl_arr})
    savemat(mainfolder+"/"+str('generator_loss.mat'),  mdict={'foo': dl_arr})

    plt.figure(1)
    plt.plot(dl_arr)
    plt.savefig(mainfolder+'/discriminator_loss.png')
    plt.figure(2)
    plt.plot(gl_arr)
    plt.savefig(mainfolder+'/generator_loss.png')

else:
    print("Testing")

    save_folder = "/media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/result/"+f_arg+"/"+s_arg+"/mask/"
    
    # Path for testing data
    test_folder_path="/media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/batches/"+s_arg+"/dysarthric/testing_batches/"
    n = len(listdir(test_folder_path))
    Gnet = torch.load(join(mainfolder,"gen_g_5_d_1_Ep_100.pth"), map_location='cpu')
    for i in range(n):
        d = loadmat(join(test_folder_path, "Batch_{}.mat".format(str(i))))
        a = torch.from_numpy(d['Feat'])
        a = Variable(a.squeeze(0).type('torch.FloatTensor'))
        enh_dys = feature_extraction(a)
        Gout = Gnet(enh_dys)
        # np.save(join(save_folder,'Test_Batch_{}.npy'.format(str(i))), Gout.cpu().data.numpy())
        savemat(join(save_folder,'Test_Batch_{}.mat'.format(str(i))),  mdict={'foo': Gout.cpu().data.numpy()})
            
