''' 사전 훈련 워드 임베딩
사전에 훈련된 워드 임베딩을 불러와서 사용가능
데이터가 적으면 문제에 특화된 임베딩 벡터를 구축할 수 없음
그래서 Word2Vec이나 Glove라는 사전 훈련된 임베딩 벡터 사용
'''

''' Word2Vec
분포 가설 : '비슷한 위치에서 등장하는 단어들은 비슷한 의미를 가진다'라는 가정
예시로 '난 수학을 공부한다'면 분포 가설에 의해 수학과 공부는 유사한 의미를 가지게 된다는 것
분산 표현은 분포 가설을 이용하여 단어들의 셋을 학습하고 단어의 의미를 분산하여 표현
즉, 단어의 의미를 사용자가 지정한 차원으로 분산하여 표현하는 것

CBOW 
Word2Vec의 방식 중 한가지
CBOW는 주변 단어들을 가지고 중간 단어를 예측하는 방법 
예측할 중심 단어를 바꿔가며 학습을 진행 

사용자가 정한 벡터 차원으로 주변 단어를 학습하여 중심 단어를 예측하는 최적의 벡터 가중치를 설정 
주변 단어로 중심 단어를 더 정확하게 맞추기 위해서 이러한 가중치들이 학습되며 각 단어의 가중치를 생성하게 된다.
주변 단어들로 학습하여 도출한 값은 확률로써 나오게 되는데 이것을 cross-entropy를 통해 
원핫벡터로 표현된 중심 단어 인덱스와 맞는지 비교하면서 loss function값을 최소화하는 방향으로 학습  

Skip-gram 
중심 단어를 가지고 주변 단어를 예측하는 방법 
CBOW는 주변 단어를 학습 데이터로 이용했기 때문에 값이 여러개가 나와 평균을 해야했지만 
Skip-gram은 중심 단어 하나를 가지고 주변 단어를 예측하기 때문에 평균을 하는 과정이 없음 
'''


import nltk
import urllib.request
import zipfile
from lxml import etree
import re
from nltk.tokenize import word_tokenize, sent_tokenize

# 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/master/ted_en-20160408.xml", filename="ted_en-20160408.xml")

# Preprocessing
targetXML = open('ted_en-20160408.xml', 'r', encoding = 'UTF8')

target_text = etree.parse(targetXML)
parse_text = '\n'.join(target_text.xpath('//content/text()'))

content_text = re.sub(r"\([^)]*\)", '', parse_text)

sent_text = sent_tokenize(content_text)
# 토큰화

normalized_text = []

normalized_text = []
for string in sent_text:
     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
     normalized_text.append(tokens)

result = []
result = [word_tokenize(sentence) for sentence in normalized_text]

print('샘플 개수 : {}'.format(len(result)))

from gensim.models import Word2Vec, KeyedVectors
model = Word2Vec(sentences = result, size = 100, window = 5, min_count = 5, workers = 4, sg = 0)
# 100차원으로 단어의 의미를 도출할 수 있는 벡터
# 앞, 뒤 5개의 단어로 학습
# 빈도가 5개 이상인 단어만 학습

model_result = model.wv.most_similar('man')
print(model_result)
# 학습이 잘 되었는지 확인

model.wv.save_word2vec_format('./eng_w2v')
# 저장
loaded_model = KeyedVectors.load_word2vec_format('./eng_w2v')
# 로드

''' Glove 
LSA : 각 단어의 빈도수를 바탕으로 행렬을 구축하여 잠재된 의미를 끌어내는 방법론 
Word2Vec : 실제값과 예측값에 대한 오차를 손실 함수를 통해 줄여나감 

LSA는 단어의 의미 유추에서 성능이 떨어지고 W2V는 원도우 크기 안에서 단어를 고려하기 때문에 전체적인 코퍼스의 정보 반영 못함 

Glove는 이러한 방법론의 한계를 보완하기 위해 두가지를 모두 사용 
Glove는 임베딩된 중심 단어와 주변 단어 벡터의 내적이 전체 코퍼스에서 동시 등장 확률이 되도록 만드는 것 
'''
# Glove 설치 안됨 ㅠㅠ
