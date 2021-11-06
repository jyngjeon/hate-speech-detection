import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

FINAL_MODEL = "./final_model_revised/"

torch.cuda.empty_cache()

model = AutoModelForSequenceClassification.from_pretrained(FINAL_MODEL)

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-hate-speech")

sentence = input("Input sentence: ")
model_input = tokenizer(sentence, return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)
outputs = model(**model_input, labels=labels)

predictions = outputs.logits.detach().numpy()
sum_exp = np.sum(np.exp(predictions))
predictions = np.exp(predictions) / sum_exp

print(predictions)
