from training import Trainer
from data_preprocessing import BirdDataset
import numpy as np
from initValues import config
import torch.utils.data as utils
import torchvision.transforms as transforms
import shutil

#shutil.rmtree("Output")

imgTransform = transforms.Compose([
    transforms.RandomResizedCrop(config.IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = BirdDataset(filePath='./birds/train/filenames.pickle',
                                            cidPath='./birds/train/class_info.pickle',
                                            dataDir='./birds/CUB_200_2011/CUB_200_2011/',
                                            embPath='./birds/train/char-CNN-RNN-embeddings.pickle',
                                            imgSize=(64,64),
                                            transform = imgTransform)

dataLoader = utils.DataLoader(
    dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False, num_workers=2
)

Gan = Trainer("Output")
#Train on Stage 1
Gan.train(dataLoader, 1)

#Training on Stage 2
# Gan.train(dataLoader, 2)

# Gan.sample("SampleOutput S1", 1)
# Gan.sample("SampleOutput S2", 2)

