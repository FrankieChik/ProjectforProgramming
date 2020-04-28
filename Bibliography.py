# My project has three tasks need to be done.
# First, to create a list to show all the txt files I will analyze in the following sections.
# Second, I what to figure out 10 major topics my data set includes.
# Third, to visualize my topic modeling results.
# Extra work: To network the topics and my documents.

import os
import gensim, nltk
from nltk.corpus import stopwords
from nltk.corpus import PlaintextCorpusReader
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from wordcloud import WordCloud, STOPWORDS
from os import walk
import numpy as ny
import csv
from pylab import *
from collections import Counter
from nltk.stem import WordNetLemmatizer, PorterStemmer


# The first task I want to do, is to create a list in which I can see the name of the files and the number of them.
# The code here is my first attempt.

# The following code direct us to data set which will be used in this project.
corpus_root = "/Users/hinmingfrankiechik/Desktop/text_files"

# I am trying to store all the file names in this list
filelist=[]
ignored_files={'.DS_Store', 'desktop.ini', 'Huovinen.dataset'}
sorted(filelist)

for root, dirs, files in os.walk(corpus_root):
    for file in files:
        if file not in ignored_files:
            filelist.append(os.corpus_root.join(file))

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

# Problem remains: I have six txt files which do not have the Authors' names in their title, which make my list looks strange.
mask = pd.to_numeric(df['Author'], errors='coerce').notnull()
df[mask] = df[mask].shift(axis=1)

df.sort_values(by=['Year', 'Author'], axis=0, ascending=True, inplace=True)
print (df)

# Now, we can print the list of files my data set has.
df.to_csv('/Users/hinmingfrankiechik/Desktop/Bibliography.csv', index = False)


stop_words = nltk.corpus.stopwords.words('english')
print(len(stop_words))

# However, the list originally in NLTK is not enough. I have tried to analyze my data set but some words like 'Author', numbers of years were included.
# Thus, I have to create an extended stop words list.
stop_words.extend ([
    'JSTOR', 'Author', 'which', 'who', 'hello', 'school', 'from', 'subject', 'edu', 'use',
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    'pp', 'data', 'al', 'vol', 'Open', 'Access', 'Accepted', 'Received', 'article', 'available' ,'License'
    'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December',
    'grateful', 'total', 'member', 'year', 'month', 'Fig', 'primarily', 'compared', 'group', 'potentially', 'based',
    'ways', 'shown', 'expression'
])
print(len(stop_words))

stemmer = nltk.stem.PorterStemmer()
lemm = WordNetLemmatizer()

texts = []
labels = []

for path, dirs, files in os.walk('/Users/hinmingfrankiechik/Desktop/text_files'):
    for file in files:
        if file not in ignored_files:
            with open(os.path.join(path, file)) as rf:
                text = rf.read()
                lemma_text=lemmatizer.lemmatize(text)
                tokenized_texts = nltk.word_tokenize(lemma_text)
                tokenized_texts_cleaned = [word for word in tokenized_texts if word.lower() not in stop_words]
                tokenized_texts_cleaned_no_punctuation = [word for word in tokenized_texts_cleaned if word.isalnum()]
                texts.append(tokenized_texts_cleaned_no_punctuation)
                labels.append(file[:-4])

id2word = gensim.corpora.Dictionary(texts)

ready_for_analysis_corpus = [id2word.doc2bow(text) for text in texts]
print(ready_for_analysis_corpus[0])

mallet_path = '/Users/hinmingfrankiechik/Downloads/mallet-2.0.8/bin/mallet'

lda_model = gensim.models.wrappers.ldamallet.LdaMallet(
mallet_path,
    corpus=ready_for_analysis_corpus,
    id2word=id2word,
    num_topics=10,
    optimize_interval=10,
    prefix='txt200x_'
)

lda_model.save("CDH598_200x.p")

topics = lda_model.show_topics(num_topics=10, num_words=20)

for topic in topics:
    print(f"Topic #{topic[0]}: {topic[1]}...\n\n")

# Now, let's visualize the topic modelling result.
# I create a function that allows me to transfer the results of each year into pandas. After that, the results shown in pandads will be visualized in mutiple bar charts.

names = ['Topic', 'Weight', 'KeyWords']
def read_table (txt_files):
    list(open(txt_files))
    df = pd.read_table(txt_files, names=names)
    del df['KeyWords']
    print(df)
    # The pandas data frame, which layouts the result of the above analysis, is created.
    # The following code will create a bar chart in which the pandas data frame can be visualized
    df.plot.bar(index, weights)
    plt.xticks(index, labels, rotation=90)
    plt.xlabel('Topic')
    plt.ylabel('Weight')
    plt.title("Topic Weights")
    plt.tight_layout()
    plt.show()
    return

df2 = read_table ('/Users/hinmingfrankiechik/Desktop/txt200_topickeys.txt')
print (df2)

# The above visualization shows us the weights of each topic. We can see the dominant topics across the entire corpus.
# What are the weights of the topic in an individual document?
# Which topic(s) are continuously dominant in every document?
# In this part of my code, I am going to see the topic weights by documents.
# The following code analyzes the occurrence of the keywords of each topic.

# Let's try to visualize the keywords in wordclouds for each topic.

