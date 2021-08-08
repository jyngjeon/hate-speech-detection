import os
from dataset_reader import *
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical

# Warning 끄기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Train Data
# Raw 데이터 읽기
dir = "../datasets/Competition_dataset/train.hate.csv"
data = read_data(dir)
data = data_frame2np_array(data)

# 라벨 나누기
x_train, y_train = split_data_into_x_and_y(data)

# y_train 원핫 벡터 변환
y_train, train_labels = change_label_2_int(y_train)
y_train = to_categorical(y_train)

# Debug
# print(f"Raw Data:\n{data}\nInputs:\n{x_train}\nLabel:\n{y_train}\n")

# Test Data
# Raw 데이터 읽기
dir = "../datasets/Competition_dataset/dev.hate.csv"
data = read_data(dir)
data = data_frame2np_array(data)

# 라벨 나누기
x_test, y_test = split_data_into_x_and_y(data)

# y_test 원핫 벡터 변환
y_test, test_labels = change_label_2_int(y_test)
y_test = to_categorical(y_test)

# Debug
# print(train_labels, test_labels)

# 텍스트 벡터화
vocab_size = 10000
sequence_length = 100
vectorize_layer = TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length
)
vectorize_layer.adapt(x_train)

# word2vec
embedding_dim = 16
# y = Embedding(
#     input_dim=vocab_size,
#     output_dim=embedding_dim
# )(vectorize_layer)
#
# # Sequential Model -> Functional API
# y = GlobalAveragePooling1D()(y)
# y = Dense(
#     16,
#     activation='relu'
# )(y)
# outputs = Dense(
#     3,
#     activation='softmax'
# )(y)
#
# model = Model(inputs=vectorize_layer, outputs=outputs)
model = Sequential([
    vectorize_layer,
    Embedding(vocab_size, embedding_dim, name="embedding"),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(
    x=x_train,
    y=y_train,
    batch_size=32,
    epochs=100,
    validation_data=(x_test, y_test)
)
