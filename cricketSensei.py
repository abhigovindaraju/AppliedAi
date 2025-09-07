import base64
import os
from google import genai
import openai
from google.genai import types
import sys


def generate(user_input):
    if model_choice == "gemini":
        client = genai.Client(
            api_key=""
            #api_key=os.environ.get("GEMINI_API_KEY"),
        )
        model = "gemini-2.5-flash"
        contents = [
            types.Content(
            role="user",
            parts=[
            types.Part.from_text(text=", ".join(user_inputs)), #Join all user inputs with a comma and space --> This way we can maintain the context for the model
            ],
            ),
            ]
        generate_content_config = types.GenerateContentConfig(
        temperature=1.0,
        thinking_config = types.ThinkingConfig(
        thinking_budget=0,
        ),
        )
        llm_response = ""
        for chunk in client.models.generate_content_stream(model=model,contents=contents, config=generate_content_config,):
            print(chunk.text, end="")
            llm_response += chunk.text
        return llm_response
    elif model_choice == "openai":
        client = openai.Client(api_key="")
        model="gpt-4"
    else:
        print("Invalid model choice.")
        return
    
    
    

if __name__ == "__main__":
    user_inputs = []
    print("Select AI model to use:")
    print("1. Gemini")
    print("2. OpenAI GPT-4")
    model_input = input("Enter 1 for Gemini or 2 for OpenAI: ").strip()
    if model_input == "1":
        model_choice = "gemini"
    elif model_input == "2":
        model_choice = "openai"
    else:
        print("Invalid selection. Exiting.")
        sys.exit(1)
    while(True):
        print("\n")
        user_input = input("Enter your question (type 'exit' or 'quit' to stop): ")
        if(user_input.lower() in ['exit', 'quit']):
            print("Exiting...")
            break
        if user_input is not "quit" and user_input is not "exit":
            user_inputs.append({"role":"user","content":user_input})
        print(user_inputs)
        llm_response = generate(user_inputs)
        user_inputs.append({"role":"assistant","content":llm_response})