from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint

from data_preprocessor import *
from keras_settings import *
from explanation_and_visualization import *

# Settings
basic_settings()  # GPU 설정, 경고 문구 수준 설정

# 기본 모델 세팅
lr = 1e-5
batch_size = 32
epochs = 100
model_name = "stemming-cnn-rnn-lstm-bn-model"

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

# 데이터 불러오기 및 전처리를 위한 세팅
train_dir = "../datasets/Competition_dataset/train.hate.csv"
test_dir = "../datasets/Competition_dataset/dev.hate.csv"
dict_dir = '../dictionary-data/custom_dict.txt'
kiwi = build_kiwi_model(dict_dir)

# Train Data
x_train, y_train = get_train_data(
    data_dir=train_dir,
    kiwi_model=kiwi,
    data_augmentation_size=5,
    data_augmentation_double_prob=0.3
)

# Test Data
x_test, y_test = get_test_data(
    data_dir=test_dir,
    kiwi_model=kiwi
)

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

# CNN
y = Conv1D(
    filters=num_filters,
    kernel_size=kernel_size,
    strides=1,
    padding='same',
    kernel_initializer='he_normal',
    kernel_regularizer=l2(1e-4)
)(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = LSTM(
    units=lstm_units,
)(y)

y = BatchNormalization()(y)
outputs = Dense(
    3,
    activation='softmax'
)(y)

model = Model(inputs=inputs, outputs=outputs)

# 모델 요약
summarize_model(model, model_name)

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
