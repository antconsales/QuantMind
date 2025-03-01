import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

from src.config import Config

tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL_NAME)
model = ORTModelForSequenceClassification.from_pretrained(
    Config.LLM_MODEL_NAME,
    provider=Config.LLM_PROVIDER
)
model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME)

model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
model.eval()

def analizza_sentiment(testo: str):
    inputs = tokenizer(
        testo, 
        return_tensors="pt", 
        truncation=True, 
        max_length=Config.MAX_INPUT_LENGTH  
    )
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    scores = torch.nn.functional.softmax(logits, dim=-1)
    
    return {
        "label": "positive" if scores[0][1] > Config.SENTIMENT_SELL_THRESHOLD else "negative",
        "score": scores[0][1].item()
    }

