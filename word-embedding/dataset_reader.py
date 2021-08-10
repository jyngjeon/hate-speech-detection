import pandas as pd
import numpy as np


def read_data(dir):
    '''
    데이터 읽어오기

    :param dir: 데이터 경로
    :return: 읽은 데이터
    '''
    data = pd.read_csv(dir)
    data.fillna("")
    return data


def data_frame2np_array(dataframe):
    '''
    데이터 프레임을 넘파이 배열로 변환

    :param dataframe: 데이터 프레임 (pandas)
    :return: 해당하는 넘파이 배열
    '''
    return dataframe.to_numpy()


def split_data_into_x_and_y(data_array):
    '''
    입력 데이터와 그에 따른 라벨로 나눔

    :param data_array: 데이터 배열 (numpy)
    :return: x_train, y_train -> 각각이 데이터 배열
    '''
    x_train, y_train = data_array[:, 0], data_array[:, 1]
    return x_train, y_train


def change_label_2_int(y_train):
    '''
    라벨을 정수로 변환함.

    :param y_train: y_train
    :return: 정수로 변환된 y_train
    '''
    # 라벨 불러오기
    label_names = np.unique(y_train)
    label_names.sort()
    label_idx = [t for t in range(len(label_names))]
    label_zip = zip(label_names, label_idx)

    # 라벨 인덱스 변환 딕셔너리
    labels = dict(label_zip)

    label_list = y_train.copy()  # Side Effect 제거
    for row_idx in range(len(y_train)):
        # 라벨
        label = label_list[row_idx]
        label_list[row_idx] = labels[label]
    return label_list, labels


if __name__ == "__main__":
    data = read_data("../datasets/Competition_dataset/train.hate.csv")
    print("데이터 헤드")
    print(data.head)
    print()
    print("데이터 요약")
    print(data.describe())
    print()
    print("데이터 타입")
    print(data.dtypes)
    print()
    np_data = data_frame2np_array(data)
    print(np_data)
