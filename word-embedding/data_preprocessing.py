import pandas
import torch
from hate_speech_dataset import HateSpeechDataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_tokenizer():
    return AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-hate-speech")


def label2id(labels):
    label_mapper = {
        "none": 0,
        "offensive": 1,
        "hate": 2
    }
    return pandas.DataFrame(data=[label_mapper[label] for label in labels]).T


def read_data(data_dir):
    raw_data = pd.read_csv(data_dir)
    raw_data.dropna()

    comments = raw_data["comments"]
    labels = raw_data["label"]
    labels = label2id(labels)

    tokenizer = get_tokenizer()
    enc_comments = tokenizer(
        comments.tolist(),
        truncation=True,
        padding=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    return HateSpeechDataset(encodings=enc_comments, labels=labels)


