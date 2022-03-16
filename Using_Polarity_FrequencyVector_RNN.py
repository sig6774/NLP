import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords

df = pd.read_csv('/Users/Moon/nlp-tutorial/tripadvisor_hotel_reviews.csv')
df = df.sample(n = 1000)
# print(df['Rating'].unique())
#
# print(df.isnull().sum())

pos = pd.read_table('/Users/Moon/nlp-tutorial/polarity/positive-words1.txt', sep = '/t', names = ['polarity', 'rate'])

pos['rate'] = 1
# print(pos)


neg = pd.read_table('/Users/Moon/nlp-tutorial/polarity/negative-words1.txt', sep = '/t', names = ['polarity', 'rate'])

neg['rate'] = -1
# print(neg)

polarity = pd.concat([pos, neg], axis = 0, ignore_index=True)
# print(polarity)

dict = polarity.set_index('polarity').T.to_dict('rate')
pol = dict[0]
# print(pol)


def preprocessing(data):
    eng = []
    for i in data:
        eng.append(re.sub('[^a-zA-Z]', ' ', i))

    stopword = set(stopwords.words('english'))
    inputs = [word for word in eng if not word in stopword]
    # 텍스트 데이터 정제
#     word_list = ' '.join(inputs).split()
#     # df['Review']말고 data로 해보자
#     word_list = list(set(word_list))
#     word_dict = {w: i for i, w in enumerate(word_list)}
#     n_class = len(word_dict)
#     # 각 단어에 고유한 인덱스 부여
#     # word_dictionary 구축

    maxlen = []
    for i in inputs:
        maxlen.append(len(i))

    quantile3 = pd.Series(maxlen).describe()
    quantile = quantile3['75%']
    # maxlen 설정

    input_x = []
    max_len = int(quantile)
    for sen in inputs:
        word = sen.split()

#         idx = [word_dict[n] for n in word]
        input_x.append(word[:max_len])

    for item in input_x:
        while len(item) < max_len:
            item.append('0')

#     # padding

    return input_x

text = preprocessing(df['Review'])

x = []
for i in text:
    idx = [pol[n] if n in pol else 0 for n in i]
    x.append(idx)
# print(idx)
# 필요한 단어와 필요없는 단어를 구분할 수 있는 기능


filtering = torch.LongTensor(x)
# print('filtering shape: ',filtering.shape)
# print(filtering)


def word_index(data):
    word_list = ' '.join(data).split()
    # df['Review']말고 data로 해보자
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    n_class = len(word_dict)

    maxlen = []
    for i in data:
        maxlen.append(len(i))

    quantile3 = pd.Series(maxlen).describe()
    quantile = quantile3['75%']
    # maxlen 설정

    input_x = []
    max_len = int(quantile)
    for sen in data:
        word = sen.split()

        idx = [word_dict[n] for n in word]
        input_x.append(idx[:max_len])

    for item in input_x:
        while len(item) < max_len:
            item.append(0)

    return torch.LongTensor(input_x), n_class, max_len

x, n_class, max_len = word_index(df['Review'])

print('x shape : ', x.shape)
print('filtering shape: ',filtering.shape)


# stack = torch.stack([filtering, filtering, filtering], dim = 2)
# # stack 텐서를 쌓아주는 함수
# print(stack.shape)
# # print(stack)
#
# c = nn.Embedding(n_class, 3)
# emb = c(x)
# print(emb.shape)
# #
#
# transpose = stack.transpose(1,2).float()
#
# multiple = torch.matmul(emb,transpose)
# # 메모리 한계로 연산진행 불가 1000개만 뽑아서 해보기
#
# print(multiple)
# 해당 단어에 있는 극성의 빈도를 추출해서 벡터를 만들고 기존의 임베딩 벡터 or BOW 벡터를 활용하여 곱하여 모델에 학습시키면 어떻게 될까?

def categorical(data):
    index = {i + 1: w for i, w in enumerate(range(len(data.unique())))}
    one_hot = []
    for i in data:
        zero = [0] * (len(data.unique()))
        # 데이터길이만큼 0으로 채워줌

        idx = index[i]
        # 해당 점수의 순서를 idx에 저장

        zero[idx] = 1
        # 0으로 채워진 리스트에 해당 점수인 부분만 1로 변경

        one_hot.append(zero)
    return torch.Tensor(one_hot)

y = categorical(df['Rating'])


class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

        n_hidden = hidden

        embedding_size = embedding

        self.C = nn.Embedding(n_class, embedding_size)

        self.rnn = nn.RNN(input_size=max_len, hidden_size=n_hidden)

        # self.rnn1 = nn.RNN(input_size = n_hidden, hidden_size = n_hidden)
        #
        # self.rnn2 = nn.RNN(input_size = n_hidden, )

        self.W = nn.Linear(max_len * n_hidden, 5)

    def forward(self, X, stack):
        n_hidden = hidden

        x = self.C(X)

        stacking = torch.stack([filtering] * n_hidden, dim=2)

        transpose = stacking.transpose(1, 2).float()
        # print(transpose.shape)
        # print(x.shape)

        multiple = torch.matmul(x, transpose)
        #         print(multiple.shape)

        output, h1 = self.rnn(multiple)
        #
        # o1, h2 = self.rnn1(output)

        outputs = output.view(-1, max_len * n_hidden)

        y_hat = self.W(outputs)

        return y_hat

hidden = 3
# hidden size에 따라 모델의 형태가 바뀜 (주의)
embedding = 3
model = RNN()
print(model)


def train_model(model, epoch, data, label, stack):
    train_loss = []
    avg_train_loss = []
    epochs = epoch
    train_x = data
    train_y = label
    stack = stack

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        #         print(stack.shape)

        output = model(train_x, stack)
        #         hat = output.squeeze()
        loss = criterion(output, train_y)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        train_loss.append(loss.item())

    return model, train_loss


epoch = 70
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-5)

model, train_loss = train_model(model, epoch, x, y, filtering)


''' 성능이 안좋은 이유 
1. 전체 데이터셋을 활용하지 않고 1000개만 랜덤 추출하여 사용 
2. 기존에 있는 감성사전을 기반으로 극성 벡터를 도출하였으므로 해당 데이터에 맞는 감성사전이 아님 (추후 해당 데이터에 맞는 감성사전 구축하기)
3. 데이터의 품질 문제 
4. 모델을 simple하게 정의 
5. 임베딩 벡터를 사전에 훈련된 벡터가 아님 
'''


''' Binary Classification 

def replace_col(row):
    if row['Rating'] >= 4:
        return 1
    else:
        return 0
y = df.apply(replace_col, axis = 1)

binary_y = torch.Tensor(list(y))
print(binary_y.shape)

class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

        n_hidden = hidden

        embedding_size = embedding
        

        self.C = nn.Embedding(n_class, embedding_size)

        self.rnn = nn.RNN(input_size=max_len, hidden_size=n_hidden)

        # self.rnn1 = nn.RNN(input_size = n_hidden, hidden_size = n_hidden)
        #
        # self.rnn2 = nn.RNN(input_size = n_hidden, )

        self.W = nn.Linear(max_len * n_hidden, 1)

    def forward(self, X, stack):
        n_hidden = hidden

        x = self.C(X)
        
        stacking = torch.stack([filtering] * n_hidden, dim=2)

        transpose = stack.transpose(1,2).float()
#         print(transpose.shape)
#         print(x.shape)
        
        multiple = torch.matmul(x, transpose)
#         print(multiple.shape)

        output, h1 = self.rnn(multiple)
        #
        # o1, h2 = self.rnn1(output)


        outputs = output.view(-1, max_len * n_hidden)

        y_hat = self.W(outputs)

        return y_hat

def train_model(model, epoch, data, label, stack):
    train_loss = []
    avg_train_loss = []
    epochs = epoch
    train_x = data
    train_y = label
    stack = stack
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
#         print(stack.shape)


        output = model(train_x, stack)
        hat = output.squeeze()
#         print(output.shape)
        loss = criterion(hat, train_y)
        
        loss.backward()
        optimizer.step() 
        
        if (epoch + 1) % 1 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        
        train_loss.append(loss.item())
        
    return model, train_loss



model = RNN()
criterion = nn.BCEWithLogitsLoss()
epoch = 100
optimizer = optim.Adam(model.parameters(), lr=0.001)

model, train_loss = train_model(model, epoch, x, d, stack)
# epoch 70으로 약 43%의 loss 


'''


