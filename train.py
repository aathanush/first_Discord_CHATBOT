import json
from nlp import tokenize,stem
with open('intents.json','r') as f:
    intents=json.load()


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

ignore_words=['?',',','.','!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set("tags"))

X_train=[]
y_train=[]
for (pattern_sentence,tag) in xy:
    bag_of_words()
