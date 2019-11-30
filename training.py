import torch, os, time, numpy as np
import torch.nn as nn
import torch.optim as optim
from initValues import config
from helper import GanHelper
from tensorboard import summary
import datetime
from torch.autograd import Variable

class Trainer(object):
    def __init__(self, outDir):
        self.currentTime = str(datetime.datetime.now().timestamp())
        self.modelDir = os.path.join(outDir, 'Model')
        self.imageDir = os.path.join(outDir, 'Image')
        self.logDir = os.path.join(outDir, 'Log')
        self.logDir = os.path.join(self.logDir, self.currentTime)

        os.makedirs(outDir)
        os.makedirs(self.modelDir)
        os.makedirs(self.imageDir)
        os.makedirs(self.logDir)
        #self.summaryWriter = summary.create_file_writer(self.logDir)

        self.maxEpoch = config.TRAIN.MAX_EPOCH
        self.batchSize = config.TRAIN.BATCH_SIZE
        #self.device = torch.cuda.set_device(0) if gpu
        self.device = config.DEVICE
        self.helper = GanHelper()

    def loadStage1(self):
        from stackgan_models import Stage1_Gen, Stage1_Dis
        netG = Stage1_Gen()
        netG.apply(self.helper.weightsInit)
        print (netG)

        netD = Stage1_Dis()
        netD.apply(self.helper.weightsInit)
        print(netD)

        if config.CUDA:
            netG.cuda()
            netD.cuda()

        return netG, netD

    def loadStage2(self):
        from stackgan_models import Stage1_Gen, Stage2_Gen, Stage2_Dis
        s1_Gen = Stage1_Gen()
        netG = Stage2_Gen(s1_Gen)
        netG.apply(self.helper.weightsInit)
        print (netG)

        netD = Stage2_Dis()
        netD.apply(self.helper.weightsInit)
        print(netD)

        if config.CUDA:
            netG.cuda()
            netD.cuda()

        return netG, netD

    def plotLoss(self,genLoss, disLoss):
        plt.figure(figsize=(10,5))
        plt.title('Generator and Discriminator Loss')
        plt.plot(genLoss, label='GEN')
        plt.plot(disLoss, label='DIS')
        
        plt.xlabel("Epoch")
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('Loss graph.png')

    def plotKLLoss(self, klLoss):
        plt.figure(figsize=(10,5))
        plt.title('KL Loss')
        plt.plot(klLoss, label='KL Loss')
        
        plt.xlabel("Epoch")
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('KL Loss graph.png')

    def train(self, loader, stage=1):
        if stage == 1:
            netG, netD = self.loadStage1()
        else:
            netG, netD = self.loadStage2()

        nz = config.Z_DIM
        batchSize = self.batchSize
        
        fixedNoise = Variable(torch.rand(batchSize, nz, device= self.device).normal_(0,1))    #change

        realLabel = Variable(torch.FloatTensor(batchSize).fill_(1))
        fakeLabel = Variable(torch.FloatTensor(batchSize).fill_(0))

        genLR = config.TRAIN.GEN_LR
        disLR = config.TRAIN.DIS_LR

        optimizerD = optim.Adam(netD.parameters(), lr = disLR, betas = (0.5, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr = genLR, betas = (0.5, 0.999))
        
        iters = 0
        genLoss = []
        disLoss = []
        klLoss = []
        
        for epoch in range(self.maxEpoch):
            for i, data in enumerate(loader, 0):

                #Prepare Training Data
                realImgCPU, textEmbedding = data
                realImg = Variable(realImgCPU)
                textEmbedding = Variable(textEmbedding)
                noise = Variable(torch.rand(batchSize, nz,  device=self.device).normal_(0,1))

                #Generate Fake Imgs
                inputs = (textEmbedding, noise)
                #print(inputs)
                _, fakeImgs, fakeMu, fakeLogvar = netG(textEmbedding, noise)
                
                #Updatae D Network
                netD.zero_grad()
                errD, errD_Real, errD_Fake, errD_Wrong = self.helper.computeDisLoss(netD, fakeImgs, realImg, fakeLabel, realLabel, fakeMu)
                errD.backward()
                optimizerD.step()

                # Update G Network
                netG.zero_grad()
                errG = self.helper.computeGenLoss(netD, fakeImgs, realLabel, fakeMu)
                klLoss = self.helper.KLLoss(fakeMu, fakeLogvar)
                errG_Total = errG + klLoss * config.TRAIN.COEFF.KL
                errG.backward()
                optimizerG.step()

                iters += 1
                #print(iters)
                # if iters == 5:
                #     break

                # if iters % 50:
                #     with self.summaryWriter.as_default():
                #         summary.scalar('D_Loss', errD.item())
                #         summary.scalar('G_Loss', errG.item())
                #         summary.scalar('KL_Loss', klLoss)

                #     inputs = (textEmbedding, fixedNoise)
                #     _, fakeImgs, _, _ = netG(inputs)

                    #Save Images

                print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t' % (epoch, self.maxEpoch, errD.item(), errG.item()))
                if iters % 100 == 0:

                    print('''[%d][%d/%d] Loss_D:%.4f  Loss_G: %.4f Loss_KL: %.4f
                     Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
                     Total Time: sec'''
                  % (count, i, len(data_loader),
                     errD.item(), errG.item(), kl_loss.data))
                    
                    genLoss.append(errG.item())
                    disLoss.append(errD.item())
                    klLoss.append(kl_loss.data)
                    _, fixedFake, _, _ = netG(textEmbedding, fixedNoise)
                    self.helper.saveImg(realImgCPU, fixedFake, iters, self.imageDir)
                

            endTime = time.time()
            print(endTime)

            if epoch % 10 == 0:
                self.helper.saveModel(netG, netD, self.modelDir, epoch)
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                   % (epoch, self.maxEpoch, errD.item(), errG.item()))

        self.helper.saveModel(netG, netD, self.modelDir, epoch)
        # self.summaryWriter.close()

    # def sample(dataDir, stage=1):
    #     if