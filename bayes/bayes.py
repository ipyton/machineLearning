from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

corpus = []
labels = []

corpus_test = []
labels_test = []

f = open('./smss/sms_spam.txt', mode='r', encoding='utf-8')
index = 0
while True:
    line = f.readline()
    if index == 0:
        index += 1
        continue
    if line:
        index += 1
        line = line.split(',')
        label = line[0]
        sentence = line[1]

        if index <= 5550:
            corpus.append(sentence)
            if "ham" == label:
                labels.append(0)
            else:
                labels.append(1)

        else:
            corpus_test.append(sentence)
            if "ham" == label:
                labels_test.append(0)
            else:
                labels_test.append(1)
    else:
        break

vectorizer = CountVectorizer()
fea_train = vectorizer.fit_transform(corpus) # input a vector of sentences and return a count vector of words

vectorizer_test = CountVectorizer(vocabulary=vectorizer.vocabulary_) # use metrics which is used in vectorizer
fea_test = vectorizer_test.fit_transform(corpus_test)

clf = MultinomialNB(alpha=1)
clf.fit(fea_train, labels)

pred = clf.predict(fea_test)
for p in pred:
    print(p)





