from kiwipiepy import Kiwi
from dataset_reader import *
from posts_reader import *

# Data Loading
dir = "../datasets/YooBeyoungWoo_dataset/hate_speech_binary_dataset.csv"
data = read_data(dir)

# Preprocessing
data = data_frame2np_array(data)
posts, _ = split_data_into_x_and_y(data)
reader = PostsReader(posts)  # Reader

kiwi = Kiwi()
result = kiwi.extract_words(
    reader=reader.read_posts,
    min_cnt=10,
    max_word_len=10,
    min_score=0.25
)
kiwi.prepare()

word_list = [res[0] for res in result]

for post in posts:
    sep, _ = kiwi.analyze(post)[0]
    for word, tag, _, _ in sep:
        if word in word_list:
            print(f"문장: {post}\n단어:{word}\n품사:{tag}")

# TODO: 2021-08-10 word를 키로 하고, 품사의 리스트를 val로 하는 딕셔너리 -> 딕셔너리에 있는 모든 품사를 다 등록