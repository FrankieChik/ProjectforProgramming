import gensim, nltk, os


texts = []
labels = []

for path, dirs, files in os.walk('/Users/hinmingfrankiechik/Desktop/text_files'):
    for file in files:
        with open(os.path.join(path, file)) as rf:
            text = rf.read()
            tokens = nltk.word_tokenize(text)
            cleaned = [word for word in tokens if word.isalnum()]
            texts.append(cleaned)
            labels.append(file[:-4])

corpus_dictionary = gensim.corpora.Dictionary(texts)

process_corpus = [corpus_dictionary.doc2bow(text) for text in texts]

number_of_topics=3

mallet_path = '/Users/hinmingfrankiechik/Downloads/mallet-2.0.8/bin/mallet'
lda_model = gensim.models.wrappers.ldamallet.LdaMallet(
mallet_path,
    corpus=process_corpus,
    id2word=corpus_dictionary,
    num_topics=number_of_topics,
    optimize_interval=3,
    prefix='txt_'
)

topics = lda_model.show_topics(num_topics=number_of_topics, num_words=10)

for topic in topics:
    print(topic)
