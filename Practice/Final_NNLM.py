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

df = pd.read_csv('/Users/Moon/nlp-tutorial/tripadvisor_hotel_reviews.csv')
df = df.sample(n = 5000)
print(df['Rating'].unique())

print(df.isnull().sum())



def replace_col(row):
    if row['Rating'] >= 4:
        return 1
    else:
        return 0
df['Rating'] = df.apply(replace_col, axis = 1)


def preprocessing(data, label):
    word_list = ' '.join(data).split()
    # df['Review']말고 data로 해보자
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    n_class = len(word_dict)
    # 각 단어에 고유한 인덱스 부여

    input_x = []
    max_len = 100
    for sen in data:
        word = sen.split()

        idx = [word_dict[n] for n in word]
        input_x.append(idx[:max_len])

    for item in input_x:
        while len(item) < max_len:
            item.append(0)
    # maxlen에 맞게 padding

    y = [i for i in label]

    return torch.LongTensor(input_x), torch.Tensor(y), n_class

x, y, n_class = preprocessing(df['Review'], df['Rating'])

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

# Model

class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()

        n_hidden = hidden
        embedding_size = embedding

        self.C = nn.Embedding(n_class, embedding_size)

        self.H = nn.Linear(300, n_hidden, bias=False)

        self.d = nn.Parameter(torch.ones(n_hidden))

        self.U = nn.Linear(n_hidden, 1, bias=False)

        self.W = nn.Linear(300, 1, bias=False)

        self.b = nn.Parameter(torch.ones(1))

    def forward(self, X):
        X = self.C(X)

        X = X.view(-1, 300)

        tanh = torch.tanh(self.d + self.H(X))

        output = self.b + self.W(X) + self.U(tanh)

        return output

    # 모델

hidden = 3
embedding = 3
model = Simple()
print('Structure of Model: \n', model)

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

# Train Function
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

# Train
batch = 128
epoch = 50

train, test, valid = create_datasets(x,y)
# 데이터셋 생성

# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()
# Define Loss Function

optimizer = optim.Adam(model.parameters(), lr=0.001)
# Define Optimizer Function


# early stopping patience;
# validation loss가 개선된 마지막 시간 이후로 얼마나 기다릴지 지정
patience = 10


model, train_loss, valid_loss = train_model(model, batch, patience, epoch)

# 이거 뭐가 잘못됨...
# 성능이 별로 loss : 43%
