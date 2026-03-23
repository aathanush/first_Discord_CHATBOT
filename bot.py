#Import all modules
import time
import discord
import os
import json
from random import randint

client=discord.Client()
replies=[]
with open("replies.txt") as file:
    replies = file.readlines()

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
    else:
        pass
        #await message.channel.send("Hello! Nice to meet you :)")

#Open the .env file
client.run(os.getenv("TOKEN"))
