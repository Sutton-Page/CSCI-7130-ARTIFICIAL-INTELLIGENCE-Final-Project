import pandas as pd
import os
import numpy as np
import shutil


train = pd.read_json('./data/train.jsonl',lines=True)
test = pd.read_json('./data/dev.jsonl',lines=True)

root_path = os.path.abspath('.')

meme_folder_path = os.path.join(root_path,'memes')


if os.path.exists(meme_folder_path) != True:

    os.mkdir(meme_folder_path)


postive_train = train.loc[train['label'] == 1]
postive_train = postive_train['img'].values
negative_train = train.loc[train['label'] == 0]
negative_train = negative_train['img'].values

postive_test = test.loc[test['label'] == 1]
postive_test = postive_test['img'].values
negative_test = test.loc[test['label'] == 0]
negative_test = negative_test['img'].values

# organize test data

test_folder_path = os.path.join(meme_folder_path,'test')

if os.path.exists(test_folder_path) != True:

    os.mkdir(test_folder_path)

negative_test_folder = os.path.join(test_folder_path,'0')
postive_test_folder = os.path.join(test_folder_path,'1')

if os.path.exists(negative_test_folder) !=True:

    os.mkdir(negative_test_folder)

if os.path.exists(postive_test_folder) != True:

    os.mkdir(postive_test_folder)


for item in postive_test:

    
    src = os.path.join('./data/', item)
    item = item.replace("img/",'')
    dest = os.path.join(postive_test_folder,item)
    shutil.copy(src,dest)


for item in negative_test:


    src = os.path.join('./data/', item)
    item = item.replace("img/",'')
    dest = os.path.join(negative_test_folder,item)
    shutil.copy(src,dest)




# organize train data

train_folder_path = os.path.join(meme_folder_path,'train')

if os.path.exists(train_folder_path) != True:

    os.mkdir(train_folder_path)


negative_train_folder = os.path.join(train_folder_path,'0')
postive_train_folder = os.path.join(train_folder_path,'1')

if os.path.exists(negative_train_folder) !=True:

    os.mkdir(negative_train_folder)

if os.path.exists(postive_train_folder) != True:

    os.mkdir(postive_train_folder)

for item in postive_train:

    
    src = os.path.join('./data/', item)
    item = item.replace("img/",'')
    dest = os.path.join(postive_train_folder,item)
    shutil.copy(src,dest)


for item in negative_train:


    src = os.path.join('./data/', item)
    item = item.replace("img/",'')
    dest = os.path.join(negative_train_folder,item)
    shutil.copy(src,dest)


    
