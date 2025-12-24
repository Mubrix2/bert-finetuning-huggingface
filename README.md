# bert-finetuning-huggingface

Project title: 
IMDb Sentiment Classification using DistilBERT

Problem Statement:
Classify movie review as positive or negative using fine-tuned transformer model.

Model Used:
- DistilBERT (encoder-only transformer)
- fine-tuned on IMDb dataset

Training Details
- Max length: 128/256
- Learning rate: 2e-5
- Epochs: 2
- Trainer API

How to Run:
pip in stall -r requirements.txt
python src/train.py
python src/inference.py