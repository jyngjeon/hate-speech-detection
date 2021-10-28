import pandas
import torch
from hate_speech_dataset import HateSpeechDataset
import pandas as pd
import numpy as np
import pandas
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_tokenizer():
    return AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-hate-speech")


def data_classification(data: pandas.DataFrame):
    '''
    데이터를 라벨에 따라 나눔.

    :param data: 원본 데이터
    :return: 데이터 분포 3항짜리 딕셔너리
    '''
    labels = ['none', 'offensive', 'hate']
    classified = dict(
        zip(
            labels,
            map(
                lambda x: data.where(data['label'] == x),
                labels
            )
        )

    )
    return classified


def over_sampling(data: pandas.DataFrame):
    '''
    데이터의 분포를 일률적이게 만들어 줌 (기대 분포: Hate, Offensive, None -> 1 : 1 : 1)

    :param data: 원본 데이터
    :return: 오버샘플링 된 데이터
    '''
    labels = ['none', 'offensive', 'hate']
    classified = data_classification(data)
    max_count = max(map(lambda x: len(x), classified.values()))
    for label in labels:
        class_data = classified[label]
        count = len(class_data)
        if count < max_count:
            sampled_data = class_data.sample(max_count - count)
            data.append(sampled_data)
    return data


def label2id(labels):
    label_mapper = {
        "none": 0,
        "offensive": 1,
        "hate": 2
    }
    return pandas.DataFrame(data=[label_mapper[label] for label in labels]).T


def read_data(data_dir, over_sample_flag = False):
    raw_data = pd.read_csv(data_dir)
    raw_data.dropna()

    if over_sample_flag:
        raw_data = over_sampling(raw_data)

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
