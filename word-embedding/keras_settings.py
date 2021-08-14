import os
import tensorflow as tf


def basic_settings():
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

    return gpus[0]