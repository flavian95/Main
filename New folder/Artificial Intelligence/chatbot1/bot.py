
import aiml

kernel = aiml.Kernel()

kernel.learn("startup.xml")
kernel.respond("LOAD AIML B")

print("AIML Chatbot started (type quit to exit)")

while True:

    user_input = input("You: ")

    if user_input.lower() == "quit":
        break

    response = kernel.respond(user_input)
    print("Bot:", response)