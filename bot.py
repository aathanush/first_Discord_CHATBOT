#Import all modules

import discord
import os
#import keep_alive
#import torch
#import random
#import json
#from model import NeuralNet
#from nlp import bag_of_words, tokenize

#initiating a discord client class
client=discord.Client()

#if the device has gpu, use it
#device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#open the questions and answers file
#with open("intents.json","r") as file:
#    intents=json.load(file)

#Specifying the path of pytorch file
#FILE = "data.pth"
#data=torch.load(FILE)
#loading all the necessary variables required for the model
#input_size=data["input_size"]
#hidden_size=data["hidden_size"]
#output_size=data["output_size"]
#all_words=data["all_words"]
#tags=data["tags"]
#model_state=data["model_state"]

#building the model
#model=NeuralNet(input_size,hidden_size,output_size).to(device)
#model.load_state_dict(model_state)
#model.eval()

@client.event
#Command executes when the bot starts functioning
async def on_ready():
    print(f"Logged in as {client.user}")

#Executes once a message is recieved
@client.event
async def on_message(message):
    #The bot should not respond to its own messages
    if message.author==client.user:
        return

    #When the bot sees a message by the user
    if message.content.lower().startswith("hello"):
        await message.channel.send("Hello! Nice to meet you :)")
    #sentence=tokenize(message.content)
    #X = bag_of_words(sentence, all_words)
    #X=X.reshape(1,X.shape[0])
    #X=torch.from_numpy()

    #output=model(X)

    #getting the predictions
    #_, predicted = torch.max(output,dim=1)
    #grabbing the entity matching with the tag
    #tag=tags[predicted.item()]

    #using softmax to check the probability of a tag being the required one
    #probability=torch.softmax(output,dim=1)
    #prob=probability[0][predicted.item()]

    #send messages if probability is > 0.75
    #if prob > 0.75:
        #for intent in intents["intents"]:
       #     if tag==intent["tag"]:
      #          await message.channel.send(f"{random.choice(intent['responses'])}")
     #       else:
     #           await message.channel.send("Sorry! I can't understand you dear :-(")
    #else:
    #    await message.channel.send("Sorry! I can't understand you dear :-(2")

#

#keep_alive.keep_alive()
#Open the .env file
client.run(os.getenv("TOKEN"))

