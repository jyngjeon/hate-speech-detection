import torch
from transformers import AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from data_preprocessing import get_tokenizer


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


def visualize_matrix(matrix):
    return \
        f'실제 \\ 판정    None    Offensive    Hate       Sum\n' + \
        f'None      |{format(matrix[0][0], "^10")}{format(matrix[0][1], "^10")}{format(matrix[0][2], "^10")}|' + \
        f'{format(sum(matrix[0]), "^10")}\n' + \
        f'Offensive |{format(matrix[1][0], "^10")}{format(matrix[1][1], "^10")}{format(matrix[1][2], "^10")}|' + \
        f'{format(sum(matrix[1]), "^10")}\n' + \
        f'Hate      |{format(matrix[2][0], "^10")}{format(matrix[2][1], "^10")}{format(matrix[2][2], "^10")}|' + \
        f'{format(sum(matrix[2]), "^10")}\n' + \
        f'Sum       |{format(sum(map(lambda x: x[0], matrix)), "^10")}' + \
        f'{format(sum(map(lambda x: x[1], matrix)), "^10")}' + \
        f'{format(sum(map(lambda x: x[2], matrix)), "^10")}|' + \
        f'{format(sum(map(lambda x: sum(x), matrix)), "^10")}'


# Constants
# Directories
ROOT_DIR = "../datasets/Custom_dataset/"
TEST_DATA = "Surin_train_dataset_na_drop_utf-8.csv"
TEST_DIR = ROOT_DIR + TEST_DATA
RESULTS_DIR = "./results/"
CP_DIR = "checkpoint-9500/"
TORCH_MODEL = RESULTS_DIR + CP_DIR

BASE_MODEL = "monologg/koelectra-base-v3-hate-speech"

torch.cuda.empty_cache()

model = AutoModelForSequenceClassification.from_pretrained(TORCH_MODEL)
tokenizer = get_tokenizer()

test_data = pd.read_csv(TEST_DIR)
test_data.dropna()
x_test = test_data["comments"]
y_test = test_data["label"]

# validity_matrix[실제][판정]: 실제 [실제]인데 판정을 [판정]으로 함.
NONE_IDX = 0
OFFENSIVE_IDX = 1
HATE_IDX = 2

validity_matrix = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
]

for test_idx in range(len(x_test)):
    inputs = tokenizer(x_test[test_idx], return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)
    outputs = model(**inputs, labels=labels)

    predict_id = np.argmax(outputs.logits.detach().numpy())
    #  real_id = tag2id(y_test[test_idx])
    real_id = y_test[test_idx]
    validity_matrix[real_id][predict_id] += 1

print(visualize_matrix(validity_matrix))
acc = (validity_matrix[0][0] + validity_matrix[1][1] + validity_matrix[2][2]) / \
      sum(map(lambda x: sum(x), validity_matrix))
print(f"Accuracy: {acc * 100:.3f}%")
