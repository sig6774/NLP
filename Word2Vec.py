from gensim.models.keyedvectors import  KeyedVectors

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary = True, limit = 200000)

print(model['apple'])

print('similarity apple and fruit : {}'.format(model.similarity('apple', 'fruit')))

print(model.most_similar(positive = ['king', 'women'], negative = ['man'], topn = 5))