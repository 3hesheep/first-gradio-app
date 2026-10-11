import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("your-username/your-model")
tokenizer = AutoTokenizer.from_pretrained("your-username/your-model")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    return str(pred)

demo = gr.Interface(fn=predict, inputs="text", outputs="text")
demo.launch()