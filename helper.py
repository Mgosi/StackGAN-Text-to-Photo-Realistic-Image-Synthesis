import numpy as np, torch, torch.nn as nn
import torch
import torchvision.utils as vutils
from torch.nn import init
from initValues import config
import os
import torchvision, torchvision.transforms as transforms
import matplotlib.pyplot as plt

class GanHelper():
    def weightsInit(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv')!=-1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data,0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)


#   def getTrainTestData(self):
#     transform = transforms.Compose(
#       [transforms.ToTensor(),
#       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     trainSet = dset.CIFAR10(root, train=True, download=True, transform=transform)
#     trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True, num_workers=2)

#     testSet = dset.CIFAR10(root, train=False, download=True, transform=transform)
#     testLoader = torch.utils.data.DataLoader(testSet, batch_size=testBatchSize, shuffle=False, num_workers=2)

#     classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#     return trainSet, trainLoader, testSet, testLoader

#   def displayImageGrid(self, imgs):
#     device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#     plt.figure(figsize=(8,8))
#     plt.axis("off")
#     plt.title("Training Images")
#     plt.imshow(np.transpose(vutils.make_grid(imgs.to(device)[:64], padding=2,normalize=True).cpu(),(1,2,0)))
    
#   def displayAnimation(self, imgList):
#     #%%capture
#     fig = plt.figure(figsize=(8,8))
#     plt.axis("off")
#     ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in imgList]
#     ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
#     HTML(ani.to_jshtml())

    def getFidScoreGraph(self, fidScores):
        plt.figure(figsize=(10,5))
        plt.title("FID Score")
        plt.plot(fidScores)
        plt.xlabel("Iterations")
        plt.ylabel("FID Score")
        plt.show()
    
    def displayLosses(self, GLosses, DLosses, DLossesFake, DLossesReal):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(GLosses,label="G")
        plt.plot(DLosses,label="D")
        plt.plot(DLossesFake,label="D Fake")
        plt.plot(DLossesReal,label="D Real")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    
    def saveModel(self, netG, netD, pathG, pathD):
        torch.save(netG.state_dict(), pathG)
        torch.save(netD.state_dict(), pathD)
        
    # def loadModel(self, path, generator = True):
    #     device = torch.device("cuda")
    #     model = generator and Generator(ngpu) or Discriminator(ngpu)
    #     model.load_state_dict(torch.load(path, map_location="cuda:0"))  # Choose whatever GPU device number you want
    #     model.to(device)
    #     return model