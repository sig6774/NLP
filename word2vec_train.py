from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

sentences = LineSentence('./word2vec_train.txt')
# Linesentence : 라이별로 하나의 문장이라고 생각하는 것

# model = Word2Vec(sentences, size = 100, window = 3, min_count = 1, iter = 1000)
#
# model.save('Basic_word2vec.model')


model = Word2Vec.load('Basic_word2vec.model')

print(model.wv.most_similar('Korea', topn = 10))

print(len(model.wv.vocab))

score, predictions = model.wv.evaluate_word_analogies('./')
