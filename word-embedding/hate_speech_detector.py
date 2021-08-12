import os
from data_preprocessor import *
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow as tf

# Warning 끄기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 사용 가능한 GPU 리스트
gpus = tf.config.experimental.list_physical_devices('GPU')

# 하나라도 있으면
if gpus:
    try:
        # 그 중 첫 번째 GPU로 설정하고, 메모리 할당을 늘릴 수 있도록 한다.
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)

# Train Data
# Raw 데이터 읽기
dir = "../datasets/Competition_dataset/train.hate.csv"
data = read_data2np_array(dir)

# 라벨 나누기
x_train, y_train = split_data_into_x_and_y(data)

# Preprocessing
# kiwi 학습
dict_dir = '../dictionary-data/custom_dict.txt'
kiwi = build_kiwi_model(dict_dir)

# 데이터 파싱
x_train = parsing_data(x_train, kiwi)

# y_train 원핫 벡터 변환
y_train = to_one_hot(y_train)

# Debug
# print(f"Raw Data:\n{data}\nInputs:\n{x_train}\nLabel:\n{y_train}\n")

# Test Data
# Raw 데이터 읽기
dir = "../datasets/Competition_dataset/dev.hate.csv"
data = read_data2np_array(dir)

# 라벨 나누기
x_test, y_test = split_data_into_x_and_y(data)

# Preprocessing
# 데이터 파싱
x_test = parsing_data(x_test, kiwi)

# y_train 원핫 벡터 변환
y_test = to_one_hot(y_test)

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
print(x_train)
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
