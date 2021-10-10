# Hate Speech Detection

웹페이지에서 텍스트 상의 혐오 표현을 단계에 따라 판별하여 필터링하는 크롬 확장 프로그램입니다. 

[KoELECTRA](https://github.com/monologg/KoELECTRA)의 혐오 표현 판별 모델에 대한 Fine Tuning을 진행하여 모델을 구축했습니다. 추가적으로 수집한 1784개의 인터넷 뉴스/커뮤니티 댓글 데이터를 Fine Tuning에 이용했습니다.

## 구성
___
```
│  .gitignore
│  README.md
│
└─word-embedding
    │  data_preprocessing.py
    │  hate_speech_dataset.py
    │  hate_speech_detector_pretrained.py
    │  load_model_and_use.py
```

`word-embedding`는 혐오 표현 판별 모델에 대한 폴더입니다.

`data_preprocessing.py`과 `hate_speech_dataset.py` 파일은 이전 LSTM 모델을 구축할 때 사용한 코드로, 데이터 호출과 전처리를 수행합니다. 

`hate_speech_detector_pretrained.py`은 사전 학습된 모델을 Fine Tuning하며, `load_model_and_use.py`은 Fine Tuning이 진행된 모델의 정확도를 데이터 레이블별로 계산하여 정리합니다. 

```
└─word-embedding
    ├─dictionary-generator
    ├─history
    ├─model-structure
    └─saved-models
```
KoELECTRA 전이학습 모델 적용 이전 개발했던 다양한 신경망 모델을 구축하여 실험했습니다. `history`, `model-structure`, `saved-models` 폴더에 관련 내용이 저장되어 있습니다. 

`dictionary-generator`는 단어 사전 구축 과정에서 사용한 코드입니다. 