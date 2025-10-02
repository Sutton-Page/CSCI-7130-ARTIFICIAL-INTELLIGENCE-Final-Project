import tensorflow as tf
import pandas as pd
import os
import numpy as np
import shutil

train= pd.read_json('./data/train.jsonl',lines=True)

# balancing the training dataset
positive = train.loc[train['label'] == 1]
negative = train.loc[train['label'] == 0]

# grabbing 1000 samples from positive and negative



positive_img = positive['img'].values

negative_img = negative['img'].values



if os.path.exists('./memes') != True:

    os.mkdir('./memes')

if os.path.exists('./memes/1') != True:

    os.mkdir('./memes/1')

if os.path.exists('./memes/0') != True:

    os.mkdir('./memes/0')

for item in positive_img:

    src = os.path.join("./data",item)
    item = item.replace("img/",'')
    dest = os.path.join("./memes/1",item)
    shutil.copy(src,dest)
    

for item in negative_img:

    src = os.path.join('./data',item)
    item = item.replace('img/','')
    dest = os.path.join('./memes/0',item)
    shutil.copy(src,dest)


