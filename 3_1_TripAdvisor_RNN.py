import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords

df = pd.read_csv('/Users/Moon/nlp-tutorial/tripadvisor_hotel_reviews.csv')
df = df.sample(n = 2000)
# 시간이 많이 걸려 전체 데이터셋으로는 못하고 2만개의 데이터 중 10%를 활용하여 랜덤 추출
print(df['Rating'].unique())
print(df.isnull().sum())

def replace_col(row):
    if row['Rating'] >= 4:
        return 1
    else:
        return 0
# rating이 4점 이상은 1 이하는 0

def preprocessing(data):

    only = []
    for i in data:
        only.append(re.sub('[^a-zA-Z]', ' ', i))

    stops = set(stopwords.words('english'))
    inputs = [word for word in only if not word in stops]

    return inputs


def batch():
    X = []
    y = []
    max_len = 100
    for sen in x:
        word = sen.split()

        input_X = [word_dict[n] for n in word]
        X.append(input_X[:100])

    for item in X:
        while len(item) < max_len:
            item.append(0)

    df['Rating'] = df.apply(replace_col, axis=1)
    y = [i for i in df['Rating']]

    return X, y

class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

        self.C = nn.Embedding(vocab_size, embedding_size)

        self.rnn = nn.RNN(input_size = embedding_size, hidden_size = hidden)

        self.W = nn.Linear(max_len * hidden, 1)

    def forward(self, x):

        x = self.C(X)

        outputs, hidden = self.rnn(x)

        outputs = outputs.view(-1, 500)
        outputs = outputs.view(-1, (max_len*hidden))


        y_hat = self.W(outputs)


        return y_hat

if __name__ == '__main__':
    hidden = 5
    embedding_size = 3
    max_len = 100

    data = preprocessing(df['Review'])
    x = [i for i in data]
    word_list = ' '.join(x).split()
    word_list = list(set(word_list))

    word_dict = {w : i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)

    model = RNN()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    X, y = batch()

    X =  torch.LongTensor(X)
    y = torch.FloatTensor(y)


    for epoch in range(1500):
        optimizer.zero_grad()

        output = model(X)

        y_hat = output.squeeze()

        loss = criterion(y_hat, y)

        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

''' 정리
목표 : RNN모델의 sequence적인 특징을 활용하여 해당 문장의 특징을 잘 추출하는지 
1. 20,400개의 데이터를 활용하여 영어만 추출하고 불용어 처리
2. 점수를 1~3 = 0, 4~5 = 1로 변환 
3. RNN모델과 word_embedding을 이용(embedding vector : 3, hidden : 5)
4. Backpropagation을 사용하여 최적의 parameters을 추정 
5. 이진분류로써 BCEW... 손실함수를 사용하여 loss추정 

결과 : 시간이 많이 걸려 데이터의 10%를 랜덤추출하여 학습을 진행했고 약 5%의 loss를 보여줌 
'''

