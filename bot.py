#Import all modules

import discord
import os
import json


client=discord.Client()



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

#Open the .env file
client.run(os.getenv("TOKEN"))

