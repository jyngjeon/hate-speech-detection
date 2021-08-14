import os
from data_preprocessor import *
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Model, Sequential
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
batch_size = 32
epochs = 60
model_name = "stemming-vectorization-dropout-deeper-model"

# 단어 벡터화를 위한 세팅
vocab_size = 10000
sequence_length = 100
embedding_dim = 16

# 드롭아웃 세팅
dropout_prob = 0.7

# Train Data
# Raw 데이터 읽기
dir = "../../datasets/Competition_dataset/train.hate.csv"
data = read_data2np_array(dir)

# 라벨 나누기
x_train, y_train = split_data_into_x_and_y(data)

# Preprocessing
# kiwi 학습
dict_dir = '../../dictionary-data/custom_dict.txt'
kiwi = build_kiwi_model(dict_dir)

# 데이터 파싱
x_train = parsing_data(x_train, kiwi)

# y_train 원핫 벡터 변환
y_train = to_one_hot(y_train)

# Debug
# print(f"Raw Data:\n{data}\nInputs:\n{x_train}\nLabel:\n{y_train}\n")

# Test Data
# Raw 데이터 읽기
dir = "../../datasets/Competition_dataset/dev.hate.csv"
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
vectorize_layer = TextVectorization(
    input_shape=(1,),
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length
)
vectorize_layer.adapt(x_train)

# word2vec
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
    Dropout(dropout_prob),
    Dense(32, activation='relu'),
    Dropout(dropout_prob),
    Dense(16, activation='relu'),
    Dropout(dropout_prob),
    Dense(3, activation='softmax')
])

# 모델 요약
model.summary()
plot_model(model, to_file=f"{model_name}.png", show_shapes=True)

model.compile(
    optimizer='adam',
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

# 체크포인트를 위한 디렉토리 생성
save_dir = os.path.join(os.getcwd(), '../saved-models')
model_name = f'{model_name}.{epochs:03d}.tf'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# FIXME: 이후 Functional API로 교체 후, Custom callback 만들 것; 현재 modelcheckpoint가 textvectorization과 호환 안 됨.
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

# model.save_weights(f"saved-models/{model_name}", save_format='tf')
