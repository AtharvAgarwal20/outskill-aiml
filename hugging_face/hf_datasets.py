from datasets import load_dataset
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

dataset = load_dataset("angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k", token=hf_token)

print(dataset)

print("\nExample from the training set:")
print(dataset["train"][0])