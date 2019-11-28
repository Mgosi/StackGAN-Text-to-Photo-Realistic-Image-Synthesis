#from training import Trainer
from data_preprocessing import BirdDataset
import numpy as np
from initValues import config
import torch.utils.data as utils


dataset = BirdDataset(filePath='./birds/train/filenames.pickle',
                                            cidPath='./birds/train/class_info.pickle',
                                            dataDir='./birds/CUB_200_2011/CUB_200_2011/',
                                            embPath='./birds/train/char-CNN-RNN-embeddings.pickle',
                                            imgSize=(64,64))
                                            
                                        
                                            
                                            
                                            #batchSize = config.TRAIN.BATCH_SIZE)
#print(trainDataset)

dataLoader = utils.DataLoader(
    dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True, num_workers=2
)