import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import re
from nltk.corpus import stopwords
import seaborn as sns

df = pd.read_csv('/Users/Moon/nlp-tutorial/tripadvisor_hotel_reviews.csv')
df = df.sample(n = 2000)
# 10%의 데이터만 랜덤하게 추출

print(df.isnull().sum())


def replace_col(row):
    if row['Rating'] >= 4:
        return 1
    else:
        return 0


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

# 데이터를 랜덤하게 뽑았기 때문에 maxlen을 3사분위수를 활용


b = basic()
print(b)

max_len = int(b)

def batch():
    X = []
    y = []

    for sen in data:
        word = sen.split()

        input_x = [word_dict[n] for n in word]
        X.append(input_x[:max_len])

    for item in X:
        while len(item) < max_len:
            item.append(0)

    df['Rating'] = df.apply(replace_col, axis = 1)
    y = [i for i in df['Rating']]

    return X, y

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.C = nn.Embedding(vocab_size, embedding_size)
        # O : [batch_size, max_len, embedding_size]

        self.lstm = nn.LSTM(input_size = embedding_size, hidden_size = hidden)
        # O : [batch_size, max_len, hidden]

        self.W = nn.Linear(max_len * hidden, 1 )
        # I : [batch_size, max_len*hidden], O : [batch_size, 1]

    def forward(self, X):
        # print(X.shape)
        # print(X[1])
        # print('training~')
        embed = self.C(X)

        hidden_state = torch.zeros(1, len(embed), hidden)
        cell_state = torch.zeros(1, len(embed), hidden)

        # outputs, (_,_) = self.lstm(X, (hidden_state, cell_state))
        outputs, (_,_) = self.lstm(embed)


        outputs = outputs.view(-1, max_len * 3)
        y_hat =self.W(outputs)


        return y_hat

if __name__ == '__main__':
    hidden = 3
    embedding_size = 5

    word_list = ' '.join(data).split()
    word_list = list(set(word_list))

    word_dict = {w : i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)

    model = LSTM()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    X, y = batch()
    X = torch.LongTensor(X)
    y = torch.FloatTensor(y)

    for epoch in range(1000):
        optimizer.zero_grad()

        outputs = model(X)

        y_hat = outputs.squeeze()
        # y_hat = [batch_size]

        loss = criterion(y_hat, y)
        # y_hat = [batch_size], y = [batch_size]

        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

''' 정리 
Using LSTM for Text Classification 
max_len을 유동적으로 사용할 수 있도록 설정함 
다른 부분은 RNN과 유사 
'''




