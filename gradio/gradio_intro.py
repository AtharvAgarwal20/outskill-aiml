import gradio as gr
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

messageHistory: list[ChatCompletionMessageParam] = [{"role": "system", "content": "You are a helpful, friendly chatbot. Answer concisely unless the user asks for detail. Use Markdown when it improves clarity."}]

def query(param):
    newMessage: ChatCompletionMessageParam = {"role": "user", "content": param}
    messageHistory.append(newMessage)
    
    completion = client.chat.completions.create(
        model="gemini-3.5-flash",
        messages=messageHistory
    )
    
    responseMessage: ChatCompletionMessageParam = {"role": "assistant", "content": completion.choices[0].message.content}
    messageHistory.append(responseMessage)
    return completion.choices[0].message.content

demo = gr.Interface(fn=query, inputs="text", outputs="text")

demo.launch()