import numpy

list_mat = [[1,2,3], [3, 6, 9], [2, 4, 6]]
print(list_mat)
m = numpy.array(list_mat)
print(m)
print(m.shape)

m1 = numpy.random.rand(3,3)
# numpy.random.rand : 0~1사이의 랜덤한 값으로 구성된 3*3 matrix
print(m1)

m2 = numpy.zeros((3,3))
# numpy.zeros : 0으로 구성된 3*3 matrix
print(m2)

matrix = numpy.loadtxt('populations.txt')
print(matrix, numpy.shape(matrix))

# Matrix Slicing

print(m[1])
# 행 추출
print(m[:,2])
# 열 추출

# numpy는 사칙연산이 모든 요소에 적용 가능
print(m+3)
print(m/3)

# Matrix와 Matrix에도 사칙연산 적용 가능
print(m + m1)
print(m * m1)
print(m / m1)

vector = [3, 7, 10]

# Matrix * Vector 가능
print(m1.dot(vector))
# .dot()을 사용해야 가능

# NLTK
import nltk

# nltk.download()
sentence = 'Hi, This is Tom. I have many cars.'
sentence = sentence.lower()
print(sentence)

tokens = nltk.word_tokenize((sentence))
print(tokens)
text = nltk.Text(tokens)
print(text)
print(len(text.tokens))
print(len(set(text.tokens)))

for token in text.vocab():
    print(token, text.vocab()[token])

text.plot()

# 토큰의 빈도수를 그래프로 표현

text.count('.')
text.count('many')
text.dispersion_plot((['.', 'many']))
# 해당 토큰의 빈도수를 보여주며 해당 토큰이 몇번째에 나오는지 보여줌

# stemming은 현재 많이 사용하지 않음, 언어의 의미를 훼손시킬 수 있기 때문에 !!!
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

tokens1 = [t for t in tokens if t not in stop]
print(tokens1)

