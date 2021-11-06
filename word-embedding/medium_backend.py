import torch
from transformers import AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from data_preprocessing import *


def id2tag(idx):
    id2tag_dict = {
        0: "none",
        1: "offensive",
        2: "hate"
    }
    return id2tag_dict[idx]


def tag2id(idx):
    tag2id_dict = {
        "none": 0,
        "offensive": 1,
        "hate": 2
    }
    return tag2id_dict[idx]


def cli():
    sentence = input("Input sentence: ")
    model_input = tokenizer(sentence, return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)
    outputs = model(**model_input, labels=labels)

    # 소프트맥스
    predictions = outputs.logits.detach().numpy()
    sum_exp = np.sum(np.exp(predictions))
    predictions = np.exp(predictions) / sum_exp

    # 예측
    predict_id = np.argmax(predictions)

    print(
        f"* 확신의 정도(None, Offensive, Hate): " +
        f"{predictions[0][0] * 100:.2f}%, {predictions[0][1] * 100:.2f}%, {predictions[0][2] * 100:.2f}%\n" +
        f"* 결론: {id2tag(predict_id)}"
    )


# Constants
# Directories
FINAL_MODEL = "./final_model_revised/"

torch.cuda.empty_cache()

model = AutoModelForSequenceClassification.from_pretrained(FINAL_MODEL)
tokenizer = get_tokenizer()

while True:
    cli()
