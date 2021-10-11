''' CNN
보통 이미지 분석에서 사용
Convolution layer : filter를 사용해서 이미지를 스캔하며 특징 추출
    output : (I - F) / S + 1
    Extract Feature to data

텍스트에서는 Filter가 각 워드를 scan한다.
필터 사이즈에 따라 scan 범위가 다름
다수의 다른 사이즈를 가진 filter들을 사용해서 도출
'''

# Preprocessing

import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
import random
import numpy as np

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  batch_first = True)

LABEL = data.LabelField(dtype = torch.float)
# 데이터 정의함

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
# 데이터를 위에서 정의한 형식대로 train,test 데이터 불러오기
train_data, valid_data = train_data.split(random_state = random.seed(SEED))
# 데이터 분리

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = "glove.6B.50d",
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)
# 해당 텍스트에 대한 벡터를 pre_trian된 glove를 통해 불러오고 최대 단어수는 25000개로 설정


BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device
)
# 각각의 데이터로 정한 BATCH_SIZE만큼 반복할 수 있도록 해주는 것

# Build Model
'''
Convolution Layer
이미지는 2차원이지만 텍스트는 보통 1차원이다.
하지만 텍스트를 2차원으로 변경할 수 있다. Word embedding을 통해 바꿀 수 있다. 
각 단어를 설명하는 벡터를 포함하는 방식 
filter는 [N*emb_size]가 된다. n은 정수로 지정이 가능
filter가 몇개의 단어를 scan할건지는 사용자가 지정하며 scan한뒤 값은 scalar로 나오며 그것을 합치면 vector가 된다.
filter를 여러개 설정하면서 filter가 다른 특징들을 학습하게 한다.

Pooling Layer
filter를 통해 나온 결과를 Max or Average해서 하나의 Scalar값으로 도출 
'''
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module): # nn.Module 상속
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        self.conv_0 = nn.Conv2d(in_channels = 1,
                                out_channels = n_filters,
                                # 필터를 통하면 필터 개수만큼의 output 도출
                                kernel_size = (filter_sizes[0], embedding_dim))

        self.conv_1 = nn.Conv2d(in_channels = 1,
                                out_channels = n_filters,
                                kernel_size = (filter_sizes[1], embedding_dim))

        self.conv_2 = nn.Conv2d(in_channels= 1,
                                out_channels= n_filters,
                                kernel_size= (filter_sizes[2], embedding_dim))

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        embedded = self.embedding(text)
        # text = [batch size, sent len]
        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]

        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))

        # conved_n = [batch size, n_filters, sent len - filter_size[n] + 1 ]
        # 이거 이해가 안됨

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        # pooled_n = [batch_size, n_filters]

        cat = self.dropout(torch.cat((pooled_0,pooled_1,pooled_2), dim = 1))

        # cat = [batch size, n_filters * len(filter_sizes)]
        # 잘모르겠음

        return self.fc(cat)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 50
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

pretrain_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrain_embeddings)
# glove의 pretrain된 데이터를 불러옴

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
# UNK_IDX라는 것은 0으로 지정하여 학습에 영향을 주지못하게 함
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

# Train Model

import torch.optim as optim

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):

    rounded_preds = torch.round(torch.sigmoid(preds))
    # 모델로 도출된 확률값을 근접한 정수로 변경
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions , batch.label)
        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'chap4-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


# Test Model
model.load_state_dict(torch.load('chap4-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test loss : {test_loss:.3f} | Test acc : {test_acc * 100:.2f}%')
