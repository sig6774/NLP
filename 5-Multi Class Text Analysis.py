'''
6 Classes 분류
TREC 데이터 셋 사용
'''

import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
import random

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm')
LABEL = data.LabelField()

train_data, test_data = datasets.TREC.splits(TEXT, LABEL, fine_grained = False)
# fine_grained :

train_data, valid_data = train_data.split(random_state = random.seed(SEED))

print(vars(train_data[-1]))

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = 'glove.6B.50d',
                 unk_init = torch.Tensor.normal_)
LABEL.build_vocab(train_data)

print(LABEL.vocab.stoi)

BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Set Iterator
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device
)

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels = 1, out_channels = n_filters, kernel_size = (fs, embedding_dim)) for fs in filter_sizes])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        text = text.permute(1, 0)
        # permute : matrix의 차원 인덱스를 통해 transpose
        # 원래 text = [sent len, batch size] permute -> text = [batch size, sent len]

        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved = [batch size, n_filters, sent len - filter_sizes[n]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim = 1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 50
N_FILTERS = 100
FILTER_SIZES = [2, 3, 4]
OUTPUT_DIM = len(LABEL.vocab)
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
# pretrain된 벡터를 불러와서 모델에 로드

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


# 최적화 함수 및 손실 함수(Optimizer Function & Loss Function)
import torch.optim as optim
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

def categorical_accuracy(preds, y):
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
'''
binary 문제와 다르게 multiple 문제임으로 모델의 output은 각 라벨의 확률값으로 도출됨
도출된 확률값에서 가장 큰 확률값을 가지는 것에 대해서 해당 라벨값을 지정해주고 그것을 추정값으로 함 
추정값과 라벨값을 비교하여 정확도를 산출 
'''

# Train
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text)
        # 모델의 추정값 도출

        loss = criterion(predictions, batch.label)
        # 배치사이즈마다 추정값과 실제값을 비교하며 loss 도출

        acc = categorical_accuracy(predictions, batch.label)

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

            predictions = model(batch.text)
            # 모델의 추정값 도출

            loss = criterion(predictions, batch.label)

            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 20

best_valid_loss = float('inf')
# float('inf') : loss를 양의 무한대로 표현해주는 것

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'chap5_model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

# Test
model.load_state_dict(torch.load('chap5_model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'Test Loss : {test_loss:.3f} | Test Acc : {test_acc*100:.2f}%')