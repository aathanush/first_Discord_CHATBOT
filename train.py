
#importing all the modules
import json
from nlp import tokenize,stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from model import NeuralNet
from torch.utils.data import Dataset, DataLoader

#loading the chat data (which is in json format)
with open('intents.json','r') as f:
    intents=json.load(f)

#all_words would consist of all the question words, tags would contain the catergory in which the question would belong
#xy contains the tag and word as a tuple (x,y)
all_words=[]
tags=[]
xy=[]

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
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

#Hyperparameters for the model
input_size = len(X_train[0])
hidden_size = 64
output_size = len(tags)
batch_size=4
learning_rate=0.001
num_epochs=1000

dataset = ChatDataset()
train_loader=DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True, num_workers=0)

#creating the model
#Use GPU if it is available
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=NeuralNet(input_size,hidden_size,output_size).to(device)

#loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop
for epoch in range(num_epochs):
    for (words,labels) in train_loader:
        words=words.to(device)
        labels=labels.to(device)
        #forward
        outputs=model(words).type(torch.LongTensor)
        loss=criterion(outputs,labels)
        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1)%100==0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')
print(f"final loss, loss={loss.item():.4f}")

#Storing the details of the model
model={"model_state":model.state_dict(), "input_size":input_size, "output_size":output_size,"hidden_size":hidden_size,"all_words":all_words,"tags":tags}

#Creating a file data.pth (PyTorch)
FILE = "data.pth"
torch.save(FILE)

#notifying that the file has been saved
print(f"training complete!! file saved to {FILE}")

