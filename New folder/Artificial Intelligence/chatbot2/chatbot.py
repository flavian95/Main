import json
from openai import OpenAI

client = OpenAI(api_key="#")

memory_file = "memory.json"

try:
    with open(memory_file, "r") as f:
        conversation = json.load(f)
except:
    conversation = []

print("AI Chatbot started (type quit to exit)")

while True:

    user_input = input("You: ")

    if user_input.lower() == "quit":
        break

    conversation.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation
    )

    reply = response.choices[0].message.content

    print("Bot:", reply)

    conversation.append({"role": "assistant", "content": reply})

    with open(memory_file, "w") as f:
        json.dump(conversation, f)