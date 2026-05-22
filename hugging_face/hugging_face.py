from dotenv import load_dotenv
from PIL import Image
from transformers import pipeline
import requests
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

classifier = pipeline("image-classification", model="timm/convit_base.fb_in1k")

url = "https://i.pinimg.com/736x/cd/f1/35/cdf1350b3603ec01affbfa933bb6e79b.jpg"
image = Image.open(requests.get(url, stream=True).raw)

predictions = classifier(image)

print("Image classification results:\n")
for prediction in predictions:
    print(f"- {prediction['label']}: {prediction['score']:.2f}")