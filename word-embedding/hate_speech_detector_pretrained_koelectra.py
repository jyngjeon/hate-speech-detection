from transformers import TFElectraModel, ElectraTokenizer, TFElectraForSequenceClassification, TensorType
from data_preprocessor import *
from keras_settings import basic_settings

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy


def label_text2int(labels):
    label_dict = {
        "none": 0,
        "offensive": 1,
        "hate": 2
    }
    return [label_dict[label] for label in labels]


# Settings
# gpu = basic_settings()
lr = 1e-5
batch_size = 16
epochs = 100

# KoELECTRA-Base
model = TFElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

# Load Data
# 데이터 불러오기 및 전처리를 위한 세팅
train_dir = "../datasets/Competition_dataset/train.hate.csv"
test_dir = "../datasets/Competition_dataset/dev.hate.csv"
# dict_dir = '../dictionary-data/custom_dict.txt'
# kiwi = build_kiwi_model(dict_dir)

# Train Data
train_data = read_data(train_dir)
x_train = train_data['comments'].tolist()
y_train = label_text2int(train_data['label'])

# x_train = [f"[CLS] {datum} [SEP]" for datum in x_train]
train_encodings = tokenizer(
    text=x_train,
    return_tensors=TensorType.TENSORFLOW,
    truncation=True,
    padding="longest",
    add_special_tokens=True
)
train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        dict(train_encodings),
        y_train
    )
)

# Test Data
test_data = read_data(test_dir)
x_test = test_data['comments'].tolist()
y_test = label_text2int(test_data['label'])

# x_test = [f"[CLS] {datum} [SEP]" for datum in x_test]
test_encodings = tokenizer(
    text=x_test,
    return_tensors=TensorType.TENSORFLOW,
    truncation=True,
    padding="longest",
    add_special_tokens=True
)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        dict(test_encodings),
        y_test
    )
)

model.compile(
    optimizer=RMSprop(learning_rate=lr),
    loss=model.compute_loss,
    metrics=['accuracy']
)

model.fit(
    train_dataset.shuffle(1000).batch(batch_size),
    epochs=epochs,
    batch_size=batch_size
)

