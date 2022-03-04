
#importing all the modules
import json
from nlp import tokenize,stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#loading the chat data (which is in json format)
with open('intents.json','r') as f:
    intents=json.load()

#all_words would consist of all the question words, tags would contain the catergory in which the question would belong
#xy contains the tag and word as a tuple (x,y)
all_words=[]
tags=[]
xy=[]

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intents['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

#ignoring all the words that are not alphanumeric
ignore_words=['?',',','.','!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))

X_train=[]
y_train=[]
for (pattern_sentence,tag) in xy:
    bag=bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)
    label=tags.index(tag)
    y_train.append(label)

X_train=np.array(X_train)
y_train=np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples=len(X_train)
        self.x_data=X_train
        self.y_data=y_train

    def __getitem__(self, idx):
        return self.x_data[idx],self.y_data[idx]

    def __len__(self):
        return self.n_samples

batch_size=8
dataset = ChatDataset()
train_loader=DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True, num_workers=2)


