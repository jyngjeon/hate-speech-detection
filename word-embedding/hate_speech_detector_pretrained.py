from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from data_preprocessor import read_data


def id2tag(idx):
    id2tag_dict = {
        0: "none",
        1: "offensive",
        2: "hate"
    }
    return id2tag_dict[idx]


print(torch.cuda.get_device_name(torch.cuda.current_device()))

# Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-hate-speech")
model = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-hate-speech")

# Constants
ROOT_DIR = "../datasets/Competition_dataset"
TRAIN_DATA = "train.hate.csv"
TEST_DATA = "dev.hate.csv"
TRAIN_DIR = ROOT_DIR + TRAIN_DATA
TEST_DATA = ROOT_DIR + TEST_DATA

# Train Data (Fine Tuning)
train_data = read_data(TRAIN_DIR)
x_train = train_data["comments"]
y_train = train_data["label"]

x_train_encodings = tokenizer(x_train, truncation=True, add_special_tokens=True, return_tensors="pt")
train_dataset = {key: torch.tensor(val[idx]) for key, val in x_train_encodings.items()}


test_data = read_data("../datasets/Competition_dataset/dev.hate.csv")
x_test = test_data["comments"]
y_test = test_data["label"]

correct_cnt = 0
data_cnt = 0
for test_idx in range(len(x_test)):
    inputs = tokenizer(x_test[test_idx], return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)
    outputs = model(**inputs, labels=labels)

    cls = np.argmax(outputs.logits.detach().numpy())
    tag = id2tag(cls)

    if tag == y_test[test_idx]:
        correct_cnt += 1
    data_cnt += 1
print(f"정확도: {correct_cnt/data_cnt * 100:.2f}%")
