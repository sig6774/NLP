import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

df = pd.read_csv('/Users/Moon/nlp-tutorial/tripadvisor_hotel_reviews.csv')
print(df['Rating'].unique())
print(df.isnull().sum())

def replace_col(row):
    if row['Rating'] >= 4:
        return 1
    else:
        return 0
# rating이 4점 이상은 1 이하는 0

df['Rating'] = df.apply(replace_col, axis = 1)


class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()

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


if __name__ == '__main__':
    embedding_size = 3
    n_hidden = 3

    word_list = ' '.join(df['Review']).split()

    word_list = list(set(word_list))

    word_dict = {w: i for i, w in enumerate(word_list)}

    n_class = len(word_dict)

    model = Simple()

    m = nn.Sigmoid()
    criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_x = []

    for sen in df['Review']:
        word = sen.split()

        input_X = [word_dict[n] for n in word]
        input_x.append(input_X[:100])
    max_len = 100

    for item in input_x:
        while len(item) < max_len:
            item.append(0)

    input_x = torch.LongTensor(input_x)
    target_y = torch.Tensor(df['Rating'])

    for epoch in range(3000):
        optimizer.zero_grad()
        output = model(input_x)


        o = output.squeeze()
        sigmoid = m(o)

        # s = []
        # for i in sigmoid:
        #     if i >= 0.5:
        #         s.append(1)
        #     else:
        #         s.append(0)
        #
        # s = torch.Tensor(s)

        loss = criterion(sigmoid, target_y)
        if (epoch + 1) % 300 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

# 와 미쳤다... 모델이 돌아갔당~~~~~~~~~
# 내가 파이토치로 해냈다리~