from kiwipiepy import Kiwi
from data_preprocessor import *
from posts_reader import *


def add_word_tag(word2tag, word, tag):
    '''
    단어에 품사 추가; 중복 불허

    :param word2tag: 단어 - 품사 매핑 딕셔너리
    :param word: 단어
    :param tag: 품사
    :return: None
    '''
    if word in word2tag.keys():
        word2tag[word].add(tag)
    else:
        word2tag[word] = {tag}


def get_column(dir, column = 0):
    '''
    데이터가 들어있는 열

    :param dir: 데이터셋을 불러올 디렉토리
    :param column: 열
    :return: 열에서 가져온 데이터
    '''
    data = read_data(dir)
    data = data_frame2np_array(data)
    posts, _ = split_data_into_x_and_y(data, column)
    return posts


def format_dictionary_line(word, tag):
    return f"{word}\t{tag}\n"


def dump_dictionary_data(word2tag, dir):
    '''
    결과를 파일로 내보냄.

    :param word2tag: 단어 - 품사 매핑 딕셔너리
    :param dir: 저장할 디렉토리
    :return:
    '''
    with open(dir, mode='w', encoding='utf-8') as f:
        for word, tags in word2tag.items():
            for tag in tags:
                dict_line = format_dictionary_line(word, tag)
                f.write(dict_line)


# Directories
dir_bin = "../datasets/YooBeyoungWoo_dataset/hate_speech_binary_dataset.csv"
dir_top = "../datasets/YooBeyoungWoo_dataset/hate_speech_topic_dataset.csv"
dir_else = "../datasets/YooBeyoungWoo_dataset/hate_speech_data.csv"

# 데이터셋 구성
dataset = np.concatenate([
    get_column(dir_bin),
    get_column(dir_top, 1),
    get_column(dir_else, 1)
])

# Reader 구성
reader = PostsReader(dataset)

# kiwi 새 단어 추출
kiwi = Kiwi()
result = kiwi.extract_words(
    reader=reader.read_posts,
    min_cnt=10,
    max_word_len=10,
    min_score=0.25
)
kiwi.prepare()

# 새 단어 품사 태깅
word_list = [res[0] for res in result]
word2tag = dict()

for post in dataset:
    # Null Exception
    if post != post:
        continue

    sep, _ = kiwi.analyze(post)[0]
    for word, tag, _, _ in sep:
        # print(f"문장: {post}\n단어:{word}\n품사:{tag}")
        if word in word_list:
            add_word_tag(word2tag, word, tag)

print("dump data")
dump_dir = "../dictionary-data/custom_dict.txt"
dump_dictionary_data(word2tag, dump_dir)
