import pickle
import numpy as np
import pandas as pd
import os
from PIL import Image
import random

def load_classID(path):
    
    with open(path,'rb') as f:
        class_ids = pickle.load(f,encoding='latin1')
    return class_ids   

def load_embedding(path):
    
    with open(path,'rb') as f:
        emb = pickle.load(f,encoding='latin1')
        emb = np.array(emb)
        
    return emb

def load_filenames(path):
    with open(path,'rb') as f:
        file_name = pickle.load(f,encoding='latin1')
    return file_name

def bounding_boxes(direc):
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

def load_image(image_path, bound, size):
    img = Image.open(image_path).convert('RGB')
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

def load_dataset(f_path, cid_path, direc, emb_path, imgSize):
    filenames = load_filenames(f_path)
    class_ids = load_classID(cid_path)
    bounding_box = bounding_boxes(direc)
    all_embeddings = load_embedding(emb_path)
    
    X, y, embd = [], [], []
    for index, filename in enumerate(filenames):
        bb=bounding_box[filename]
        img_name = '{}/images/{}.jpg'.format(direc, filename)
        img = load_image(img_name, bb, imgSize)
        
        all_embeddings1 = all_embeddings[index, :, :]

        embedding_ix = random.randint(0, all_embeddings1.shape[0] - 1)
        embedding = all_embeddings1[embedding_ix, :]
        X.append(np.array(img))
        y.append(class_ids[index])
        embd.append(embedding)
        
    X= np.array(X)
    y= np.array(y)
    embd = np.array(embd)
    
    return X,y,embd


# Loading training and testing datasets

# X_train, y_train, embd_train = load_dataset(f_path='birds/train/filenames.pickle',
#                                             cid_path='birds/train/class_info.pickle',
#                                             direc='birds/CUB_200_2011/CUB_200_2011/',
#                                             emb_path='birds/train/char-CNN-RNN-embeddings.pickle',
#                                             imgSize=(64,64))

# X_test, y_test, embd_test = load_dataset(f_path='birds/test/filenames.pickle',
#                                          cid_path='birds/test/class_info.pickle', 
#                                          direc='birds/CUB_200_2011/CUB_200_2011/', 
#                                          emb_path='birds/test/char-CNN-RNN-embeddings.pickle', imgSize=(64,64))