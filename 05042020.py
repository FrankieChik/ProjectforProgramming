import os, string, gensim, nltk
from nltk.corpus import stopwords, PlaintextCorpusReader
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize, RegexpTokenizer
import pandas as pd
import numpy as np
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

# This page of code create several data frames to show the results of my topic modelling.

loading = LdaModel.load('/Users/hinmingfrankiechik/Desktop/CDH598_200.p')
print(loading.print_topics(num_topics=10, num_words=20))

corpus='/Users/hinmingfrankiechik/Desktop/text_files'

ignored_files={'.DS_Store', 'desktop.ini', 'Huovinen.dataset'}
# I am trying to store all the file names in this list
filelist=[]
sorted(filelist)

for path, dirs, files in os.walk(corpus):
    dirs.sort()
    for dir in dirs:
        print(os.path.join(path, dir))
    for file in files:
        if file not in ignored_files:
            print(os.path.join(path, dir, file))
            filelist.append(os.path.join(file))

# Print all the files in al
for name in filelist:
    print(name)

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
