import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
import random

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm')

LABEL = data.LabelField(dtype = torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

print(f'Number of train example : {len(train_data)}')
print(f'Number of test example : {len(test_data)}')

print(vars(train_data.examples[0]))

train_data, valid_data = train_data.split(random_state = random.seed(SEED))
# default로 사용해서 데이터 분리

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

MAX_VOCAB_SIZE = 25000

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE, min_freq = 2)
LABEL.build_vocab(train_data)

# print(len(TEXT.vocab))
# print(len(LABEL.vocab))
#
#
# print(TEXT.vocab.stoi)
# # 현 단어 집합에서 단어와 맵핑된 고유한 정수 출력
# print(TEXT.vocab.itois[:10])
# # 현 단어 집합에서 고유한 정수와 맵핑된 단어 출력

print(LABEL.vocab.stoi)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# iterator
# iterator를 통해 학습 및 평가를 반복, 각 반복마다 배치사이즈만큼
# BucketIterator는 유사한 길이를 가지는 데이터를 함께 Batchsize로 지정해주는 iterator
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device
)

# Model
# nn.Modul과 super를 사용해서 RNN구현

import torch.nn as nn

class RNN(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):

        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        # one-hot벡터인 input_dim을 입력으로 받아서 Dense-vector로 변환
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        # RNN을 거치면서 Hidden_dim으로 도출
        self.fc = nn.Linear(hidden_dim, output_dim)
        # init에서 모델의 layers를 정의

    def forward(self, text):

        embedded = self.embedding(text)

        output, hidden = self.rnn(embedded)

        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        # ? 0,1이라서 -1을 하는 곤감???

        return self.fc(hidden.squeeze(0))



INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 150
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model) : ,} trainable parameters')

import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr = 1e-3)

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):

    rounded_preds = torch.round(torch.sigmoid(preds))
    # torch.round : 0.12123 -> 0, 0.9566 -> 1
    # rounded_preds는 확률로 도출된 것을 라벨로 변경해줌
    correct = (rounded_preds == y).float()
    # 정수로 변경된 라벨을 float으로 변경
    acc = correct.sum() / len(correct)

    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)
        # Dimension을 줄여줌 1차원으로 변경

        loss = criterion(predictions, batch.label)
        # 추정값과 실제값을 비교하면서 loss 도출

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        # item()을 이용해서 텐서로부터 스칼라를 추출
        # 즉 고유한 값을 추출

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


# training
EPOCH = 10
best_valid_loss = float('inf')

for epoch in range(EPOCH):
    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'chap1_model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


# Evaluation

model.load_state_dict(torch.load('chap1_model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

