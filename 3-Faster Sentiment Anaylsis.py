# FastText 사용
# FastText : n-gram의 인풋 문장을 계산하고 문장 끝에 추가?
# n-gram이 핵심일듯

# Bi-Gram

def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x
# 한칸 띄어서 토큰 두개씩 중복되지 않고 넣어줌

print(generate_bigrams(['This', 'film', 'is', 'terrible']))

import torch
from torchtext.legacy import data
from torchtext.legacy import datasets

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  preprocessing = generate_bigrams)

LABEL = data.LabelField(dtype = torch.float)

import random
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state = random.seed(SEED))

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = "glove.6B.50d",
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)


# iterator 생성


BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device
)

import torch.nn as nn
import torch.nn.functional as F

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):

        super().__init__()
        # nn.Module의 기능을 상속 받음

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        # self.embedding은 nn.Embedding으로 정의됨
        # nn.Embedding은 vocab_size, embedding_dim, padding으로 구성

        self.fc = nn.Linear(embedding_dim, output_dim)
        # self.fc는 embedding_dim을 input으로 output_dim을 output으로 내뱉는 것

    def forward(self, text):

        embedded = self.embedding(text)
        # self.embedding은 nn.Embedding으로 단어 개수, 임베딩 벡터 차원, 패딩 유무로 구성됨

        # output = [sent_len, batch_size, embedding_dim]

        embedded = embedded.permute(1, 0 ,2)
        # permute 차원을 재구성함

        # output = [batch_size, sent_len, embedding_dim]

        pooled = F.avg_pool2d(embedded, (embedded.shape[1],1)).squeeze(1)
        # squeeze로 차원 줄임

        # pooled = [batch_size, embedding_dim]

        return self.fc(pooled)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 50
OUTPUT_DIM = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)

pretrained_embeddings = TEXT.vocab.vectors
# TEXT에 있는 벡터를 가져옴
model.embedding.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
# unk와 pad를 제로 벡터로 구성하여 학습에 영향을 주지 않도록 설정

# Train Model

import torch.optim as optim

optimizer = optim.Adam(model.parameters())
critertion = nn.BCEWithLogitsLoss()

model = model.to(device)
critertion = critertion.to(device)

def binary_accuracy(preds, y):

    rounded_preds = torch.round(torch.sigmoid(preds))
    # 소수점 값을 0,1로 변환
    correct = (rounded_preds == y).float()
    # 맞는것들 실수ㅗㄹ
    acc = correct.sum() / len(correct)
    # 최종적인 Accuracy

    return acc

def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        # 학습마다 가중치값 초기화
        predictions = model(batch.text).squeeze(1)

        loss = critertion(predictions, batch.label)
        # 미리 지정했던 손실함수로 loss 구함

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_acc += acc.item()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# 학습을 통해 통합 로스값과 정확도값을 도출

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            loss = critertion(predictions, batch.label)
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

# Training

N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, critertion)

    valid_loss, valid_acc = evaluate(model, valid_iterator, critertion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'chap3_model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

''' 느낀점 
Layer가 적고 Embedding을 계산해서 Linear Layer에 넣었으므로 연산이 단순하다
연산이 단순하기 때문에 빠른 학습이 가능
'''

