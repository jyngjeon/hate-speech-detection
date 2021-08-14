import os

from data_preprocessor import *
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import GRU, LSTM, SimpleRNN
from tensorflow.keras.layers import Conv1D, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
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

# Settings
# 기본 모델 세팅
lr = 1e-5
batch_size = 32
epochs = 100
model_name = "stemming-lstm-bn-model"

# 단어 벡터화를 위한 세팅
vocab_size = 10000
sequence_length = 100
embedding_dim = 16

# CNN 세팅
num_filters = 64
kernel_size = 3

# RNN 세팅
gru_units = 256
lstm_units = 256
simple_rnn_units = 128

# 드롭아웃 세팅
dropout_prob = 0.7

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

# 데이터 증식
x_train, y_train = data_augmentation(
    x_data=x_train,
    y_data=y_train,
    size=5,
    double_prob=0.3
)

# Debug
# print(f"Raw Data:\n{data}\nInputs:\n{x_train}\nLabel:\n{y_train}\n")
print(len(x_train), len(y_train))

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

# 텍스트 벡터화 레이어 정의
vectorize = TextVectorization(
    input_shape=(1,),
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length
)
vectorize.adapt(x_train)
vectorize.adapt(x_test)

# 입력 레이어
inputs = Input(shape=(1,), dtype=tf.string, name="text inputs")

y = vectorize(inputs)

# word2vec
y = Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim
)(y)

# # CNN
# y = Conv1D(
#     filters=num_filters,
#     kernel_size=kernel_size,
#     strides=1,
#     padding='same',
#     kernel_initializer='he_normal',
#     kernel_regularizer=l2(1e-4)
# )(y)
# y = BatchNormalization()(y)
# y = Activation('relu')(y)
#
# y = Conv1D(
#     filters=num_filters//2,
#     kernel_size=kernel_size,
#     strides=1,
#     padding='same',
#     kernel_initializer='he_normal',
#     kernel_regularizer=l2(1e-4)
# )(y)
# y = BatchNormalization()(y)
# y = Activation('relu')(y)
#
# RNN
# y = GRU(
#     units=gru_units,
#     return_sequences=True
# )(y)
#
# y = SimpleRNN(units=simple_rnn_units, name="rnn")(y)
y = LSTM(
    units=lstm_units,
    kernel_regularizer=l2(1e-4)
)(y)

# DNN
# y = GlobalAveragePooling1D(name="pooling")(y)
y = BatchNormalization()(y)
outputs = Dense(
    3,
    activation='softmax'
)(y)

model = Model(inputs=inputs, outputs=outputs)

# 모델 요약
model.summary()
plot_model(model, to_file=f"model-structure/{model_name}.png", show_shapes=True)

model.compile(
    optimizer=RMSprop(learning_rate=lr),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

# 체크포인트를 위한 디렉토리 생성
save_dir = os.path.join(os.getcwd(), 'saved-models')
model_name = f'{model_name}.{epochs:03d}.tf'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# FIXME: Custom callback 만들 것; 현재 modelcheckpoint가 textvectorization과 호환 안 됨.
# 모델 체크포인트
checkpoint = ModelCheckpoint(
    filepath=filepath,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
)

callbacks = [checkpoint]

# 최적화
model.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    shuffle=True,
    epochs=epochs,
    validation_data=(x_test, y_test),
)

model.save_weights(f"saved-models/{model_name}", save_format='tf')
