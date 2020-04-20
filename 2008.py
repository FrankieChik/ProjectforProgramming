import gensim, nltk, os
from nltk.corpus import stopwords
from stop_words import get_stop_words


ignore_files = {'.DS_Store', '.txt'}

texts = []
labels = []

for path, dirs, files in os.walk('/Users/hinmingfrankiechik/Desktop/text_files/2008'):
    for file in files:
        if file not in ignore_files:
            with open(os.path.join(path, file)) as rf:
                text = rf.read()
                tokens = nltk.word_tokenize(text)
                cleaned = [word for word in tokens if word.isalnum()]
                tokens_without_sw=[word for word in tokens if not word in stopwords.words()]
                texts.append(cleaned)
                labels.append(file[:-4])

corpus_dictionary = gensim.corpora.Dictionary(texts)

processed_corpus = [corpus_dictionary.doc2bow(text) for text in texts]
print(processed_corpus[0])

number_of_topics=5

mallet_path = '/Users/hinmingfrankiechik/Downloads/mallet-2.0.8/bin/mallet'

lda_model = gensim.models.wrappers.ldamallet.LdaMallet(
mallet_path,
    corpus=processed_corpus,
    id2word=corpus_dictionary,
    num_topics=number_of_topics,
    optimize_interval=10,
    prefix='txt2008_'
)

lda_model.save("CDH598_2008.p")

topics = lda_model.show_topics(num_topics=number_of_topics, num_words=20)

for topic in topics:
    print(f"Topic #{topic[0]}: {topic[1]}...\n\n")

