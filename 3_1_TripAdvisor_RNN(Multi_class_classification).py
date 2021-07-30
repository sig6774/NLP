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

# def replace_col(row):
#     if row['Rating'] >= 4:
#         return 1
#     else:
#         return 0
# rating이 4점 이상은 1 이하는 0

def categorical(data):
    a = {i+1: w for i,w in enumerate(range(len(data.unique())))}
    b = []
    for i in df['Rating']:
        one_hot = [0]*(len(df['Rating'].unique()))
        index = a[i]
        one_hot[index] = 1
        b.append(one_hot)
    return b
# Multi_class

def preprocessing(data):
    only = []
    for i in data:
        only.append(re.sub('[^a-zA-Z]', ' ', i))

    stops = set(stopwords.words('english'))
    inputs = [word for word in only if not word in stops]

    return inputs

data = preprocessing(df['Review'])

def basic():
    b = []
    for i in data:
        b.append(len(i))


    des = pd.Series(b).describe()
    return des['75%']

b = basic()
print(b)

max_len = int(b)
hidden = 5

def batch():
    X = []
    y = []

    for sen in x:
        word = sen.split()

        input_X = [word_dict[n] for n in word]
        X.append(input_X[:max_len])

    for item in X:
        while len(item) < max_len:
            item.append(0)


    y = [i for i in categorical(df['Rating'])]

    return X, y

class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

        self.C = nn.Embedding(vocab_size, embedding_size)

        self.rnn = nn.RNN(input_size = embedding_size, hidden_size = hidden)

        self.W = nn.Linear(max_len * hidden, 5)

    def forward(self, x):

        x = self.C(X)

        outputs, hidden = self.rnn(x)

        outputs = outputs.view(-1, max_len*5)
        # outputs = outputs.view(-1, max_len*hidden) 이거 안되네


        y_hat = self.W(outputs)


        return y_hat

if __name__ == '__main__':
    hidden = 5
    embedding_size = 3


    data = preprocessing(df['Review'])
    x = [i for i in data]
    word_list = ' '.join(x).split()
    word_list = list(set(word_list))

    word_dict = {w : i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)

    model = RNN()

    criterion = nn.MultiLabelSoftMarginLoss()
    # 정답이 원핫인코딩으로 구현되어있고 추정값이 다차원의 확률값으로 나올때 사용하는 Loss Function
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    X, y = batch()

    X =  torch.LongTensor(X)
    y = torch.FloatTensor(y)


    for epoch in range(700):
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
1. max_len을 사분위수를 활용하여 추출할 수 있도록 활용 
2. max_len을 기준으로 각 단어에 대해 고유한 embedding vector 사용
3. RNN모델과 word_embedding을 이용(embedding vector : 3, hidden : 5)
4. Backpropagation을 사용하여 최적의 parameters을 추정 
5. MultiClass Classification이므로 nn.MultiLabelSoftMarginLoss() 사용 



추가 : 5개의 라벨을 분류하는 Task임에도 불구하고 epoch 700에서 2%의 loss값을 도출 

batch_size를 정하고 validation set을 정해 값을 도출할 예정 
'''

