#Import all modules

import discord
import os
import keep_alive
import torch
import random
import json
from model import NeuralNet
from nlp import bag_of_words, tokenize
client=discord.Client()

#if the device has gpu, use it
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#open the questions and answers file

model=NeuralNet(input_size,hidden_size,output_size).to(device)

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

    #When the bot sees the message hello
    if message.content.lower().startswith("hello"):
        await message.channel.send("Hello! Nice to meet you :)")


keep_alive.keep_alive()
#Open the .env file
client.run(os.getenv("TOKEN"))

