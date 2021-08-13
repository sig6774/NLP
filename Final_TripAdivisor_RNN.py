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
# df = df.sample(n = 1000)
print(df['Rating'].unique())

print(df.isnull().sum())


def preprocessing(data):
    eng = []
    for i in data:
        eng.append(re.sub('[^a-zA-Z]', ' ', i))

    stopword = set(stopwords.words('english'))
    inputs = [word for word in eng if not word in stopword]
    # 텍스트 데이터 정제

    word_list = ' '.join(inputs).split()
    # df['Review']말고 data로 해보자
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    n_class = len(word_dict)
    # 각 단어에 고유한 인덱스 부여
    # word_dictionary 구축

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

        idx = [word_dict[n] for n in word]
        input_x.append(idx[:max_len])

    for item in input_x:
        while len(item) < max_len:
            item.append(0)

    # padding

    return torch.LongTensor(input_x), n_class, max_len


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

x, n_class, max_len = preprocessing(df['Review'])
print(n_class)
print(x)
print(max_len)

y = categorical(df['Rating'])
print(y)

batch_size = 128


def create_datasets(data, label):
    valid_size = 0.2

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=410, shuffle=True)
    # train과 test 데이터셋 분리
    train_data = TensorDataset(train_x, train_y)
    test_data = TensorDataset(test_x, test_y)

    num_train = len(train_data)
    idx = list(range(num_train))
    # train_data 개수만큼 리스트 생성

    np.random.shuffle(idx)
    # suffle

    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = idx[split:], idx[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # validation set 구축과정

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(test_data, batch_size=batch_size, )
    # train, valid, test dataset

    return train_loader, test_loader, valid_loader

train, test, valid = create_datasets(x,y)


class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

        n_hidden = hidden

        embedding_size = embedding

        self.C = nn.Embedding(n_class, embedding_size)

        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=n_hidden)

        self.W = nn.Linear(max_len * n_hidden, 5)

    def forward(self, X):
        n_hidden = hidden

        x = self.C(X)

        output, h1 = self.rnn(x)


        outputs = output.view(-1, max_len * n_hidden)

        y_hat = self.W(outputs)

        return y_hat

hidden = 64
embedding = 256
model = RNN()
print(model)

# Structure of Model

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    # Early Stopping
    # 이건 그대로 가져옴


def train_model(model, batches, patience, epoch):
    train_loss = []
    valid_loss = []
    avg_train_loss = []
    avg_valid_loss = []
    batch_size = batches
    epochs = epoch
    patience = patience
    # loss 값을 저장할 리스트 생성

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, epochs + 1):

        model.train()
        for batch, (data, target) in enumerate(train, 1):
            # Mini batch를 위해 중첩 for문 사용

            optimizer.zero_grad()

            output = model(data)

            output = output.squeeze()
            # target값과 맞추기 위해 차원 생성
            #             print(output)

            loss = criterion(output, target)
            # loss값 도출

            #             print(loss)
            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())

        #             print(train_loss)

        model.eval()
        for data, target in valid:
            output = model(data)
            output = output.squeeze()

            loss = criterion(output, target)
            valid_loss.append(loss.item())

        train_loss1 = np.average(train_loss)
        valid_loss1 = np.average(valid_loss)
        avg_train_loss.append(train_loss1)
        avg_valid_loss.append(valid_loss1)
        # 1epoch의 batch마다 나오는 loss들을 평균을 구함

        epoch_len = len(str(epochs))

        print_msg = (
                    f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' + f'train_loss: {train_loss1 :.5f} ' + f'valid_loss: {valid_loss1 :.5f}')

        print(print_msg)

        #         train_losses = []
        #         valid_losses = []

        # early_stopping는 validation loss가 감소하였는지 확인이 필요하며,
        # 만약 감소하였을경우 현제 모델을 checkpoint로 만든다.
        early_stopping(valid_loss1, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # best model이 저장되어있는 last checkpoint를 로드한다.
    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, avg_train_loss, avg_valid_loss

# 같은 이름의 변수 여러개 중복해서 사용하지 않기!!!

batch = 128
epoch = 20
patience = 5

train, test, valid = create_datasets(x,y)
# 데이터셋 생성

# criterion = nn.BCELoss()
criterion = nn.MultiLabelSoftMarginLoss()
# Define Loss Function

optimizer = optim.Adam(model.parameters(), lr=1e-5)
# Define Optimizer Function


# early stopping patience;
# validation loss가 개선된 마지막 시간 이후로 얼마나 기다릴지 지정



model, train_loss, valid_loss = train_model(model, batch, patience, epoch)

fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

# validation loss의 최저값 지점을 찾기
minposs = valid_loss.index(min(valid_loss))+1
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 0.5) # 일정한 scale
plt.xlim(0, len(train_loss)+1) # 일정한 scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Loss : 43%

# 빈도수를 가중치로 줘서 학습해보기

# rnn의 마지막 hidden state 정보를 통해서 결과를 도출해보면 어떤 결과가 나올려나