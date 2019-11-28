import torch, os, time, numpy as np
import torch.nn as nn
import torch.optim as optim
from initValues import config
from helper import GanHelper
from tensorboard import summary
import datetime

class Trainer(object):
    def __init__(self, outDir):
        self.currentTime = str(datetime.datetime.now().timestamp())
        self.modelDir = os.path.join(outDir, 'Model')
        self.imageDir = os.path.join(outDir, 'Image')
        self.logDir = os.path.join(outDir, 'Log')
        self.logDir = os.path.join(self.logDir, self.currentTime)
        os.makedirs(self.modelDir)
        os.makedirs(self.imageDir)
        os.makedirs(self.logDir)
        self.summaryWriter = summary.create_file_writer(self.logDir)

        self.maxEpoch = config.TRAIN.MAX_EPOCH
        self.batchSize = config.TRAIN.BATCH_SIZE
        #self.device = torch.cuda.set_device(0) if gpu
        self.device = config.DEVICE
        self.helper = GanHelper()

    def loadStage1(self):
        from stackgan_models import Stage1_Gen, Stage1_Dis
        netG = Stage1_Gen()
        netG.apply(self.helper.weightsInit())
        print (netG)

        netD = Stage1_Dis()
        netD.apply(self.helper.weightsInit())
        print(netD)

        if config.CUDA:
            netG.cuda()
            netD.cuda()

        return netD, netG

    def loadStage2(self):
        from stackgan_models import Stage1_Gen, Stage2_Gen, Stage2_Dis
        s1_Gen = Stage1_Gen()
        netG = Stage2_Gen(s1_Gen)
        netG.apply(self.helper.weightsInit())
        print (netG)

        netD = Stage2_Dis()
        netD.apply(self.helper.weightsInit())
        print(netD)

        if config.CUDA:
            netG.cuda()
            netD.cuda()

        return netD, netG

    def train(self, loader, stage=1):
        if stage == 1:
            netG, netD = self.loadStage1()
        else:
            netG, netD = self.loadStage2()

        nz = config.Z_DIM
        batchSize = self.batchSize
        
        fixedNoise = torch.rand(batchSize, nz, device= self.device).normal_(0,1)    #change

        realLabel = torch.FloatTensor(batchSize).fill_(1)
        fakeLabel = torch.FloatTensor(batchSize).fill_(0)

        genLR = config.TRAIN.GENERATOR_LR
        disLR = config.TRAIN.DISCRIMINATOR_LR

        optimizerD = optim.Adam(netD.parameters(), lr = disLR, betas = (0.5, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr = genLR, betas = (0.5, 0.999))
        
        iters = 0

        for epoch in range(self.maxEpoch):
            #add lr Decay is neeeded

            for i, data in enumerate(loader, 0):

                #Prepare Training Data
                realImg, textEmbedding = data.to(self.device)
                noise = torch.rand(batchSize, nz,  device=self.device)

                #Generate Fake Imgs
                inputs = (textEmbedding, noise)
                _, fakeImgs, fakeMu, fakeLogvar = netG(inputs)
                
                #Updatae D Network
                netD.zero_grad()
                errD, errD_Real, errD_Fake, errD_Wrong = self.helper.computeDisLoss(netD, fakeImgs, realImg, fakeLabel, realLabel, fakeMu)
                errD.backward()
                optimizerD.step()

                #Update G Network
                netG.zero_grad()
                errG = self.helper.computeGenLoss(netD, fakeImgs, realLabel, fakeMu)
                klLoss = self.helper.KLLoss(fakeMu, fakeLogvar)
                errG_Total = errG + klLoss * config.TRAIN.COEFF.KL
                errG_Total.backward()
                optimizerG.step()

                iters += 1

                if iters % 50:
                    with self.summaryWriter.as_default():
                        summary.scalar('D_Loss', errD.data[0])
                        summary.scalar('G_Loss', errG.data[0])
                        summary.scalar('KL_Loss', klLoss.data[0])

                    inputs = (textEmbedding, fixedNoise)
                    _, fakeImgs, _, _ = netG(inputs)

                    #Save Images

            endTime = time.time()
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                   % (epoch, self.maxEpoch, errD.item(), errG.item()))

        self.helper.saveModel(netG, netD, self.modelDir, epoch)
        self.summaryWriter.close()

x = Trainer("./")
x.loadStage2()