import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict(text):
    tokenizer = AutoTokenizer.from_pretrained("models/distilbert-imdb")
    model = AutoModelForSequenceClassification.from_pretrained(
        "models/distilbert-imdb"
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs, dim=1).item()

    return "Positive" if label == 1 else "Negative"

if __name__ == "__main__":
    text = "This movie was surprisingly good"
    print(predict(text))
