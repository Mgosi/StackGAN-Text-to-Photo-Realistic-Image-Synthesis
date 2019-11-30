import torch
import torch.nn as nn
from initValues import config
from torch.autograd import Variable

class ConBlock(nn.Module):
  def __init__(self, inpC, outC, kernel_size = 4, stride = 2, padding = 1, bias = False, BN = True, leaky = False):
    super(ConBlock, self).__init__()
    self.BN = BN
    self.leaky = leaky
    self.conv = nn.Conv2d(inpC, outC, kernel_size= kernel_size, stride = stride, padding = padding, bias = bias)
    self.batchNorm = nn.BatchNorm2d(outC)
    self.relu = nn.ReLU(inplace=True)
    self.leakyRelu = nn.LeakyReLU(0.2, inplace = True)

  def forward(self, x):
    out = self.conv(x)
    if self.BN:
      out = self.batchNorm(out)
    out = self.leakyRelu(out) if self.leaky else self.relu(out)
    return out

def upBlock(inPlanes, outPlanes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        ConBlock(inPlanes, outPlanes, kernel_size = 3, stride = 1)

    )
    return block

class getLogits(nn.Module):
  def __init__(self, ndf, nef, bCondition=True):
    super(getLogits, self).__init__()
    self.dfDim = ndf
    self.efDim = nef
    self.bCondition = bCondition

    if bCondition:
      self.logits = nn.Sequential(
        ConBlock(ndf * 8 + nef, ndf * 8, kernel_size=3, stride=1, leaky=True),
        ConBlock(ndf * 8, 1, stride=4, BN = False),
        nn.Sigmoid()
      )
    else:
      self.logits = nn.Sequential(
        ConBlock(ndf * 8, 1, stride=4, BN = False),
        nn.Sigmoid()
      )
  
  def forward(self, inpH, inpC=None):
    if self.bCondition and inpC is not None:
      inpC = inpC.view(-1, self.efDim, 1, 1)
      inpC = inpC.repeat(1, 1, 4, 4)
      inpHC = torch.cat((inpH, inpC), 1)
    else:
      inpHC = inpH

    out = self.logits(inpHC)
    return out.view(-1)


class ResBlock(nn.Module):
    def __init__(self, channelNum):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            ConBlock(channelNum, channelNum, kernel_size=3, stride=1),
            ConBlock(channelNum, channelNum, kernel_size=3, stride=1, leaky=False),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class CA_NET(nn.Module):
  def __init__(self):
    super(CA_NET,self).__init__()
    self.tDim = config.TEXT.DIMENSION
    self.cDim = config.GAN.CONDITION_DIM
    self.fc = nn.Linear(self.tDim, self.cDim * 2, bias=True)
    self.relu = nn.ReLU()

  def encode(self, text_embedding):
    x = self.relu(self.fc(text_embedding))
    mu = x[:, :self.cDim]
    logvar = x[:, self.cDim:]
    return mu, logvar

  def reparametrize(self, mu, logvar):
    std = logvar.mul(0.5).exp_()
    if config.CUDA:
      eps = torch.cuda.FloatTensor(std.size()).normal_()
      #torch.cuda.
    else:
      eps = torch.FloatTensor(std.size()).normal_()
    eps = Variable(eps)
    return eps.mul(std).add_(mu)

  def forward(self, text_embedding):
     mu, logvar = self.encode(text_embedding)
     condEmb = self.reparametrize(mu, logvar)
     return condEmb, mu, logvar


class Stage1_Gen(nn.Module):
  def __init__(self):
    super(Stage1_Gen, self).__init__()
    self.gfDim = config.GAN.GF_DIM * 8
    self.efDim = config.GAN.CONDITION_DIM
    self.zDim = config.Z_DIM
    self.defineModel()

  def defineModel(self):
    inp = self.zDim + self.efDim
    ngf = self.gfDim
    self.caNet = CA_NET()
    self.net = nn.Sequential(
        nn.Linear(inp, ngf * 4 * 4, bias = False),
        nn.BatchNorm1d(ngf * 4 * 4),
        nn.ReLU(True)
    )
    self.upSam = nn.Sequential(
        upBlock(ngf, ngf // 2),
        upBlock(ngf // 2, ngf // 4),
        upBlock(ngf // 4, ngf // 8),
        upBlock(ngf // 8, ngf // 16)
    )
    self.img = nn.Sequential(
        
        nn.Conv2d(ngf // 16, 3, kernel_size=3, stride=1, padding=1, bias=False),
        nn.Tanh()
    )
  
  def forward(self, textEmbedding, noise):
    condEmb , mu, logvar = self.caNet(textEmbedding)
    zCondEmb = torch.cat((noise, condEmb), 1)
    catImgEmb = self.net(zCondEmb)
    catImgEmb = catImgEmb.view(-1, self.gfDim, 4, 4)
    catImgEmb = self.upSam(catImgEmb)
    fakeImg = self.img(catImgEmb)
    return None, fakeImg, mu, logvar

class Stage1_Dis(nn.Module):
  def __init__(self):
    super(Stage1_Dis, self).__init__()
    self.dfDim = config.GAN.DF_DIM
    self.efDim = config.GAN.CONDITION_DIM
    self.defineModel()

  def defineModel(self):
    ndf, nef = self.dfDim, self.efDim

    self.encodeImg = nn.Sequential(
        ConBlock(3, ndf, BN=False, leaky = True),
        #(ndf) x 32 x 32
        ConBlock(ndf, ndf*2, leaky = True),
        #(ndf*2) x 16 x 16
        ConBlock(ndf*2, ndf * 4, leaky = True),
        #(ndf*4) x 8 x 8
        ConBlock(ndf * 4, ndf * 8, leaky = True)
        #(ndf*8) x 4 x 4
    )

    self.getCondLogits = getLogits(ndf, nef)

  def forward(self, image):
    imgEmbedding = self.encodeImg(image)
    return imgEmbedding

class Stage2_Gen(nn.Module):
  def __init__(self, Stage1_Gen):
    super(Stage2_Gen, self).__init__()
    self.gfDim = config.GAN.GF_DIM
    self.efDim = config.GAN.CONDITION_DIM
    self.zDim = config.Z_DIM
    self.stage1Gen = Stage1_Gen

    for param in self.stage1Gen.parameters():
      param.requires_grad = False
    self.defineModel()

  def _make_layer(self, block, channelNum):
    layers = []
    for i in range(config.GAN.RES_NUM):
      layers.append(block(channelNum))
    return nn.Sequential(*layers)

  def defineModel(self):
    ngf = self.gfDim

    self.caNet = CA_NET()

    self.encoder = nn.Sequential(
        ConBlock(3, ngf, kernel_size=3, stride = 1, BN = False),
        ConBlock(ngf, ngf * 2),
        ConBlock(ngf * 2, ngf * 4)
    )

    self.hrJoint = nn.Sequential(
        ConBlock(self.efDim + ngf * 4, ngf * 4, kernel_size = 3, stride = 1)
    )

    self.residual = self._make_layer(ResBlock, ngf * 4)

    self.upSam = nn.Sequential(
        upBlock(ngf * 4, ngf * 2),
        upBlock(ngf * 2, ngf),
        upBlock(ngf, ngf // 2),
        upBlock(ngf // 2, ngf // 4)
    )

    self.img = nn.Sequential(
        nn.Conv2d(ngf // 4, 3, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.Tanh()
    )

  def forward(self, textEmbedding, noise):
    _, stage1_Img, _, _ = self.Stage1_Gen(textEmbedding, noise)
    stage1_Img = stage1_Img.detach()
    encodedImg = self.encoder(stage1_Img)

    condEmb , mu, logvar = self.caNet(textEmbedding)
    condEmb = condEmb.view(-1, self.efDim, 1, 1)
    condEmb = condEmb.repeat(1, 1, 16, 16)
    catImgEmb = torch.cat([encodedImg, condEmb], 1)
    catImgEmb = self.hrJoint(catImgEmb)
    catImgEmb = self.residual(catImgEmb)

    catImgEmb = self.upSam(catImgEmb)

    fakeImg = self.img(catImgEmb)
    return stage1_Img, fakeImg, mu, logvar

class Stage2_Dis(nn.Module):
  def __init__(self):
    super(Stage2_Dis, self).__init__()
    self.dfDim = config.GAN.DF_DIM
    self.efDim = config.GAN.CONDITION_DIM
    self.defineModel()

  def defineModel(self):
    ndf, nef = self.dfDim, self.efDim
    self.encodeImg = nn.Sequential(
        ConBlock(3, ndf, BN = False, leaky = True),
        #128 * 128 * (ndf)
        ConBlock(ndf, ndf * 2, leaky = True),
        #64 x 64 x (ndf*2)
        ConBlock(ndf * 2, ndf * 4, leaky = True),
        #32 x 32 x (ndf*4)
        ConBlock(ndf * 4, ndf * 8, leaky = True),
        #16 x 16 x (ndf*8)
        ConBlock(ndf * 8, ndf * 16, leaky = True),
        #8 x 8 x (ndf*16)
        ConBlock(ndf * 16, ndf * 32, leaky = True),
        #4 x 4 x (ndf*32)
        ConBlock(ndf * 32, ndf * 16, kernel_size = 3, stride=1, leaky = True),
        #4 x 4 x ndf * 16
        ConBlock(ndf * 16, ndf * 8, kernel_size = 3, stride=1, leaky = True)
        #4 x 4 x ndf x 8
    )

    self.condLogits = getLogits(ndf, nef)
    self.uncondLogits = getLogits(ndf, nef, bCondition = False)

  def forward(self, image):
    imgEmbedding = self.encodeImg(image)
    return imgEmbedding