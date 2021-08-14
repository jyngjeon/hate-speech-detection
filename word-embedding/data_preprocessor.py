import pandas as pd
import numpy as np
from kiwipiepy import Kiwi
from tensorflow.keras.utils import to_categorical

# Constants
valid_tag_list = [
    "NNG", "NNP", "VV", "VA", "VCP", "VCN", "SW",
    "MM", "MAG", "IC", "XPN", "XR", "SN", "UN"
]


def read_data(dir):
    '''
    데이터 읽어오기

    :param dir: 데이터 경로
    :return: 읽은 데이터
    '''
    data = pd.read_csv(dir)
    data.dropna()
    return data


def data_frame2np_array(dataframe):
    '''
    데이터 프레임을 넘파이 배열로 변환

    :param dataframe: 데이터 프레임 (pandas)
    :return: 해당하는 넘파이 배열
    '''
    return dataframe.to_numpy()


def read_data2np_array(dir):
    return data_frame2np_array(read_data(dir))


def split_data_into_x_and_y(data_array, x_column=0, y_column=1):
    '''
    입력 데이터와 그에 따른 라벨로 나눔

    :param data_array: 데이터 배열 (numpy)
    :param x_column: x를 가져올 열
    :param y_column: y를 가져올 열
    :return: x_train_raw, y_train -> 각각이 데이터 배열
    '''
    x_train, y_train = data_array[:, x_column], data_array[:, y_column]
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


def concatenate_data(*data):
    dataset = [*data]
    return np.concatenate(dataset)


def build_kiwi_model(dict_dir):
    '''
    kiwi 모델을 반환합니다.

    :param dict_dir: 딕셔너리 디렉토리
    :return: kiwi
    '''
    kiwi = Kiwi()
    kiwi.load_user_dictionary(dict_dir)
    kiwi.prepare()
    return kiwi


def interpret_kiwi_analysis(kiwi_output):
    '''
    kiwi의 분석 결과에서 단어들만 뽑아내어, 띄어쓰기로 이어진 문자열을 만듭니다.

    :param kiwi_output: kiwi model의 분석 결과 (tuple의 리스트)
    :param do_data_augmentation: 데이터 augmentation 여부
    :return: 띄어쓰기로 이어진 문자열
    '''
    tag_filter = filter(lambda x: x[1] in valid_tag_list, kiwi_output)
    return ' '.join(map(lambda x: x[0], tag_filter))


def parsing_data(data, kiwi):
    '''
    구성 된 kiwi로 형태소 분석을 진행합니다.

    :param data: 원본 데이터
    :param kiwi: kiwi model
    :param do_data_augmentation: data augmentation 여부
    :return: 형태소 분석이 완료된 데이터
    '''
    # 분석 및 결과 np array 출력
    analysis = [kiwi.analyze(datum) for datum in data]
    return np.array([*map(lambda x: interpret_kiwi_analysis(x[0][0]), analysis)])


def parse_data_for_tokenizing(data):
    return np.array(["[CLS] "+post+"[sep]" for post in data])


def data_augmentation(x_data, y_data, size, double_prob):
    '''
    문장에서 단어를 하나씩 뺌으로써 데이터량을 늘린다.
    확률적으로 단어가 2개 빠지기도 한다.
    문장 길이가 size보다 작을 경우, 그 문장은 증식되지 아니한다.

    :param x_data: 입력 데이터 전체
    :param y_data: 라벨 데이터 전체
    :param size: 불릴 데이터량 (배수)
    :param double_prob: 두 개가 빠질 확률
    :return:
    '''
    augmented_data = np.copy(x_data)
    augmented_label = np.copy(y_data)

    for x_idx in range(x_data.shape[0]):
        x = x_data[x_idx].split()
        y = y_data[x_idx]

        if len(x) <= size:
            continue

        del_idx = np.random.random_integers(
            low=0,
            high=len(x) - 1,
            size=size
        )
        for idx in del_idx:
            x_tmp = x.copy()
            del x_tmp[idx]

            if np.random.random(1)[0] < double_prob:
                del_idx_2 = np.random.randint(
                    low=0,
                    high=len(x) - 2,
                    size=size
                )[0]
                del x_tmp[del_idx_2]

            x_tmp = ' '.join(x_tmp)
            augmented_data = np.concatenate([augmented_data, np.array([x_tmp])])
            augmented_label = np.concatenate([augmented_label, np.array([y])])

        if x_idx % 100 == 99:
            print(f"{(x_idx / len(x_data)) * 100:.3f}% 완료")

    return augmented_data, augmented_label


def to_one_hot(data):
    data, _ = change_label_2_int(data)
    return to_categorical(data)


def get_train_data(
        data_dir, kiwi_model, data_augmentation_size, data_augmentation_double_prob
):
    # 데이터 읽기
    loaded_data = read_data2np_array(data_dir)

    # 라벨 나누기
    x_train, y_train = split_data_into_x_and_y(loaded_data)

    # Preprocessing
    # 데이터 파싱
    x_train = parsing_data(x_train, kiwi_model)

    # 원 핫 인코딩
    y_train = to_one_hot(y_train)

    # 데이터 증식
    x_train, y_train = data_augmentation(
        x_data=x_train,
        y_data=y_train,
        size=data_augmentation_size,
        double_prob=data_augmentation_double_prob
    )
    return x_train, y_train


# TODO: 과연 이게 필요할까?
def get_train_data_with_tokenizer(
    data_dir, kiwi_model, tokenizer, data_augmentation_size, data_augmentation_double_prob
):
    # 데이터 읽기
    loaded_data = read_data2np_array(data_dir)

    # 라벨 나누기
    x_train, y_train = split_data_into_x_and_y(loaded_data)

    # Preprocessing
    # 데이터 파싱 및 토크나이즈
    x_train = parsing_data(x_train, kiwi_model)

    # 원 핫 인코딩
    y_train = to_one_hot(y_train)

    # 데이터 증식
    x_train, y_train = data_augmentation(
        x_data=x_train,
        y_data=y_train,
        size=data_augmentation_size,
        double_prob=data_augmentation_double_prob
    )
    return x_train, y_train


def get_test_data(
        data_dir, kiwi_model
):
    # 데이터 읽기
    loaded_data = read_data2np_array(data_dir)

    # 라벨 나누기
    x_test, y_test = split_data_into_x_and_y(loaded_data)

    # Preprocessing
    # 데이터 파싱
    x_test = parsing_data(x_test, kiwi_model)

    # 원 핫 인코딩
    y_test = to_one_hot(y_test)

    return x_test, y_test


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
