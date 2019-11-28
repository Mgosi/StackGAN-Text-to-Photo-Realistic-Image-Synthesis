import pickle
import numpy as np
import pandas as pd
import os
from PIL import Image
import random
import torch.utils.data as data

class BirdDataset(data.Dataset):
    def __init__(self, filePath, cidPath, dataDir, embPath, imgSize, transform=None ):
        self.transform = transform
        self.imgSize = imgSize
        self.dataDir =  dataDir
        self.filenames = self.loadFilenames(filePath)
        self.classID = self.loadClassID(cidPath)
        self.embeddings = self.loadEmbedding(embPath)
        self.boundingBoxes = self.getBoundingBoxes(dataDir)

    def loadClassID(self, path):
        
        with open(path,'rb') as f:
            class_ids = pickle.load(f,encoding='latin1')
        return class_ids   

    def loadEmbedding(self, path):
        
        with open(path,'rb') as f:
            emb = pickle.load(f,encoding='latin1')
            emb = np.array(emb)
            
        return emb

    def loadFilenames(self, path):
        with open(path,'rb') as f:
            file_name = pickle.load(f,encoding='latin1')
        return file_name

    def getBoundingBoxes(self, direc):
        bb_path=os.path.join(direc,'bounding_boxes.txt')
        filename_path = os.path.join(direc, 'images.txt')
        df_bb = pd.read_csv(bb_path, delim_whitespace=True, header=None).astype(int)
        df_fn = pd.read_csv(filename_path, delim_whitespace=True, header=None)
        file_names = df_fn[1].tolist()
        dic = {img[:-4]: [] for img in file_names[:2]}
        for i in range(0, len(file_names)):
            bounding_boxes = df_bb.iloc[i][1:].tolist()
            img_key = file_names[i][:-4]
            dic[img_key]=bounding_boxes
            
        return dic    

    def loadImage(self, imagePath, bound, size):
        img = Image.open(imagePath).convert('RGB')
        wid, height = img.size
        if bound is not None:
            R = int(np.maximum(bound[2], bound[3]) * 0.75)
            center_x = int((2 * bound[0] + bound[2]) / 2)
            center_y = int((2 * bound[1] + bound[3]) / 2)
            y1 = np.maximum(0, center_y - R)
            y2 = np.minimum(height, center_y + R)
            x1 = np.maximum(0, center_x - R)
            x2 = np.minimum(wid, center_x + R)
            img = img.crop([x1, y1, x2, y2])
        img.resize(size)
        return img

    def __getitem__(self, index):
        fileName = self.filenames[index]
        bb=self.boundingBoxes[fileName]

        imgName = '{}/images/{}.jpg'.format(direc, filename)
        img = self.loadImage(imgName, bb, imgSize)
        
        embValues = self.embeddings[index, :, :]

        embeddingIX = random.randint(0, embValues.shape[0] - 1)
        embedding = embValues[embeddingIX, :]

        return img, embedding
    def __len__(self):
        return len(self.filenames)
