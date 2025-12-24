from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

def main():
    dataset = load_dataset("imdb", split="test")

    tokenizer = AutoTokenizer.from_pretrained("models/distilbert-imdb")
    model = AutoModelForSequenceClassification.from_pretrained(
        "models/distilbert-imdb"
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized = dataset.map(tokenize, batched=True)

    trainer = Trainer(model=model, tokenizer=tokenizer)
    results = trainer.evaluate(eval_dataset=tokenized)

    print(results)

if __name__ == "__main__":
    main()
