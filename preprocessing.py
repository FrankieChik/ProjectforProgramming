from os import walk
import csv
from pylab import *
from collections import Counter, defaultdict
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.util import ngrams

# This page of code preprocesses the corpus and run the topic modelling.

corpus='/Users/hinmingfrankiechik/Desktop/text_files'

# I am trying to store all the file names in this list
filelist=[]
ignored_files={'.DS_Store', 'desktop.ini', 'Huovinen.dataset'}
sorted(filelist)

texts = []
labels = []

stop_words = nltk.corpus.stopwords.words('english')
print(len(stop_words))

stop_words.extend ([
    'jstor', 'author', 'which', 'who', 'hello', 'school', 'from', 'subject', 'edu', 'use',
    'pp', 'data', 'al', 'vol', 'Open', 'access', 'accepted', 'received', 'article', 'available', 'license'
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
    'grateful', 'total', 'member', 'year', 'month', 'fig', 'primarily', 'compared', 'group', 'potentially', 'based',
    'ways', 'shown', 'expression', 'Table', 'low', 'found', 'study', 'called', 'journal', 'two', 'usa', 'et', 'new', 'also',
    'table', 'type', 'page', 'see', 'using', 'oral', 'role', 'used', 'manuscript', 'figure', 'read'
])
print(len(stop_words))

porter_stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


for path, dirs, files in os.walk('/Users/hinmingfrankiechik/Desktop/text_files'):
    dirs.sort()
    for dir in dirs:
        print(os.path.join(path, dir))
    for file in files:
        if file not in ignored_files:
            print(os.path.join(path, dir, file))
            with open(os.path.join(path, file)) as rf:
                text = rf.read()
                tokenized_texts = nltk.word_tokenize(text)
                lowercase_texts = [word.lower() for word in tokenized_texts]
                len_nonum_tokenized_texts=[word for word in lowercase_texts if len(word) > 2]
                tokenized_texts_cleaned = [word for word in len_nonum_tokenized_texts if not word in stop_words]
                lemma_texts = [lemmatizer.lemmatize(word) for word in tokenized_texts_cleaned]
                tokenized_texts_cleaned_no_punctuation = [word for word in lemma_texts if word.isalnum()]
                nonum_tokenized_texts=[word for word in tokenized_texts_cleaned_no_punctuation if not word.isnumeric()]
                texts.append(nonum_tokenized_texts)
                labels.append(file[:-4])

bigram = gensim.models.Phrases(texts, min_count=5, threshold=1)
trigram = gensim.models.Phrases(bigram[texts], threshold=1)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

texts = [bigram_mod[word] for word in texts]
texts = [trigram_mod[bigram_mod[word]] for word in texts]


id2word = gensim.corpora.Dictionary(texts)
id2word.filter_extremes(no_below=5)
ready_for_analysis_corpus = [id2word.doc2bow(text) for text in texts]
print(ready_for_analysis_corpus[0])

mallet_path = '/Users/hinmingfrankiechik/Downloads/mallet-2.0.8/bin/mallet'

lda_model = gensim.models.wrappers.ldamallet.LdaMallet(
mallet_path,
    corpus=ready_for_analysis_corpus,
    id2word=id2word,
    num_topics=10,
    optimize_interval=10,
    topic_threshold=0.0,
    prefix='txt200_'
)

lda_model.save("CDH598_200.p")
topics = lda_model.show_topics(num_topics=10, num_words=20, formatted=False)

for topic in topics:
    print(f"Topic #{topic[0]}: {topic[1]}...\n\n")
