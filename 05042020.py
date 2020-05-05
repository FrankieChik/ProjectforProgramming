import os, string, gensim, nltk
from nltk.corpus import stopwords, PlaintextCorpusReader
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize, RegexpTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from os import walk
import csv
from pylab import *
from collections import Counter, defaultdict
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from gensim.models import LdaModel, Phrases
from gensim.models.phrases import Phraser
from nltk.util import ngrams
import warnings
import seaborn as sns
from decimal import *

# The first task I want to do, is to create a list in which I can see the name of the files and the number of them.
# The code here is my first attempt.
# The following code direct us to data set which will be used in this project.

corpus='/Users/hinmingfrankiechik/Desktop/text_files'

# I am trying to store all the file names in this list
filelist=[]
ignored_files={'.DS_Store', 'desktop.ini', 'Huovinen.dataset'}
sorted(filelist)

for path, dirs, files in os.walk(corpus):
    dirs.sort()
    for dir in dirs:
        print(os.path.join(path, dir))
    for file in files:
        if file not in ignored_files:
            print(os.path.join(path, dir, file))
            filelist.append(os.path.join(file))

# Print all the files in all subdirectories.
for name in filelist:
    print(name)

# Unfortunately, the list comes out are not in chronological order.
# What I am going to do, instead, is to sort out how many files there are in each subdirectory.
# Now, I am going to transform the above result into pandas data frame.

df=pd.DataFrame(filelist, columns=['Tent'])
index = df.index
index.name = "Num"
df[['Author', 'Year', 'Title']] = df.Tent.str.split("_-_", expand=True)
del df['Tent']

print("\n After adding three new columns and deleting the original column : \n", df)

# Problem remains: I have six txt files which do not have the Authors' names in their title, which make my list looks strange.
mask = pd.to_numeric(df['Author'], errors='coerce').notnull()
df[mask] = df[mask].shift(axis=1)

df.sort_values(by=['Year', 'Author'], axis=0, ascending=True, inplace=True)
print (df)

# Now, we can print the list of files my data set has.
df.to_csv('/Users/hinmingfrankiechik/Desktop/Bibliography.csv', index = False)

pub_records = df['Year'].value_counts().sort_index(ascending=True)
print(pub_records)
pub_records.plot.bar(x='Year', y='Number')
plt.title("Number of Publications by Years")
plt.tight_layout()
plt.show()

# Proprocessing the corpus.
texts = []
labels = []

stop_words = nltk.corpus.stopwords.words('english')
print(len(stop_words))

porter_stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

for path, dirs, files in os.walk('/Users/hinmingfrankiechik/Desktop/text_files'):
    dirs.sort()
    for file in files:
        if file not in ignored_files:
            with open(os.path.join(path, file)) as rf:
                text = rf.read()
                tokenized_texts = nltk.word_tokenize(text)
                lowercase_texts = [word.lower() for word in tokenized_texts]
                len_nonum_tokenized_texts=[word for word in lowercase_texts if len(word) > 2]
                tokenized_texts_cleaned = [word for word in len_nonum_tokenized_texts if not word in stop_words]
                lemma_texts=[lemmatizer.lemmatize(word) for word in tokenized_texts_cleaned]
                stem_texts=[stemmer.stem(word) for word in lemma_texts]
                tokenized_texts_cleaned_no_punctuation = [word for word in stem_texts if word.isalnum()]
                nonum_tokenized_texts=[word for word in tokenized_texts_cleaned_no_punctuation if not word.isnumeric()]
                texts.append(nonum_tokenized_texts)
                labels.append(file[:-4])

bigram = gensim.models.Phrases(texts, min_count=5, threshold=1)
trigram = gensim.models.Phrases(bigram[texts], threshold=1)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

texts = [bigram_mod[word] for word in texts]
texts = [trigram_mod[bigram_mod[word]] for word in texts]

new_texts=[]
for i in texts:
    if isinstance(i, list):
        new_texts.extend(i)
    else:
        new_texts.append(i)

fdist = FreqDist(new_texts)
fdist.most_common(500)
df = pd.DataFrame(list(fdist.most_common(500)),columns = ['column1','column2'])
print(df)

stop_words.extend ([
    'jstor', 'author', 'which', 'who', 'hello', 'school', 'from', 'subject', 'edu', 'use',
    'pp', 'data', 'al', 'vol', 'Open', 'access', 'accepted', 'received', 'article', 'available',
    'license', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september',
    'october', 'november', 'december', 'grateful', 'total', 'member', 'year', 'month', 'fig', 'primarily',
    'compared', 'group', 'potentially', 'based', 'ways', 'shown', 'expression', 'Table', 'low', 'found',
    'study', 'called', 'journal', 'two', 'usa', 'et', 'new', 'also', 'table', 'type', 'page', 'see', 'using',
    'oral', 'role', 'used', 'manuscript', 'figure', 'read', 'one', 'two', 'used',
    'different', 'http', 'however', 'found', 'well', 'many', 'thus', 'within', 'could',
    'new', 'known', 'three', 'first', 'either', 'even', 'since', 'given', 'common', 'site', 'involved'
])
print(len(stop_words))

new_corpus=[word for word in texts if not word in stop_words]

id2word = gensim.corpora.Dictionary(new_corpus)
id2word.filter_extremes(no_below=5)
ready_for_analysis_corpus = [id2word.doc2bow(text) for text in new_corpus]
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

# I am going to visualize the result.
# We can see the topic weights in each documents and the dominant topic in a certain document.

with open('/Users/hinmingfrankiechik/Desktop/txt200_doctopics.txt','r', encoding='utf8') as rf:
    topic_info = rf.read().split("\n")
    topic_info = [item for item in topic_info if item != ""]
print(topic_info)

doc_topic_info={}

for name, line in zip(filelist, topic_info):
    new_info = line.split("\t")
    doc_topic_info[name] = [float(weight) for weight in new_info[2:]]
weights = doc_topic_info
print(doc_topic_info)

# The weights contain some exponential numbers. So I need to float them in my pandas datafrmae.
pd.options.display.float_format = '{:.5f}'.format
df=pd.DataFrame(doc_topic_info)
# transpose the rows and columns in the df.
df_transposed=df.T
df_transposed.columns=['Topic_0', 'Topic_1', 'Topic_2', 'Topic_3',
                       'Topic_4', 'Topic_5', 'Topic_6', 'Topic_7',
                       'Topic_8', 'Topic_9'
                       ]
# Adding a new column which shows the dominant topic of each document by by counting the maximum value in a row.
df_transposed['Dominant_Topic'] = np.argmax(df_transposed.values, axis=1)
df_transposed.head()
print(df_transposed)
# Save the result. But the csv.file would not float the exponential number if I do not add "float_format='%.5f'".
df_transposed.to_csv('/Users/hinmingfrankiechik/Desktop/Topic_Weights_in_Documents.csv',
                     index = True, float_format='%.5f'
                     )
sorted_df = df_transposed['Dominant_Topic'].value_counts().sort_index(ascending=True)
print(sorted_df)
sorted_df.plot.bar(x='Topic', y='Dominant_in_Documents')
plt.title("Dominant Topics in Documents")
plt.tight_layout()
plt.show()

data_flat = [w for word_list in texts for w in word_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])
df = pd.DataFrame(out, columns=['word', 'topic_id', 'weights', 'word_count'])
df.to_csv('Term_weights.csv')

warnings.filterwarnings("ignore")
lda = gensim.models.wrappers.ldamallet.LdaMallet.load('/Users/hinmingfrankiechik/Desktop/CDH598_200n.p')

fiz = plt.figure(figsize=(20, 40))
for i in range(10):
    df = pd.DataFrame(lda.show_topic(i), columns=['term', 'prob']).set_index('term')
    plt.subplot(5, 2, i + 1)
    plt.title('Topic ' + str(i + 1))
    sns.barplot(x='prob', y=df.index, data=df, label='Cities', palette='Blues_d')
    xticks(rotation=90)
    plt.xlabel('Probability')

plt.savefig('CDH598.png', dpi=100)
plt.show()
