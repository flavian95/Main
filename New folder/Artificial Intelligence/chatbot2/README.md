
AI Chatbot – OpenAI API Implementation

Description
This chatbot is implemented using the OpenAI API and Python. 
It generates responses using a large language model instead of predefined rules.

Requirements
Python 3.x
OpenAI Python library

Installation

1. Install the OpenAI library:

pip install openai

2. Obtain an OpenAI API key from:
https://platform.openai.com/api-keys

3. Replace the placeholder in chatbot.py:

client = OpenAI(api_key="YOUR_API_KEY")

Running the Chatbot

Navigate to the chatbot folder:

cd chatbot2_ai

Run the chatbot:

python chatbot.py

Type messages in the terminal to interact with the chatbot.
Type "quit" to exit.

Memory and Personalization

The chatbot stores conversation history in the file:

memory.json

This file contains the dialogue between the user and the assistant in JSON format.

Each time the chatbot runs, it loads the previous conversation from memory.json, allowing it to remember past interactions and provide personalized responses.

Technologies Used

Python
OpenAI API
JSON-based memory storage