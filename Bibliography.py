# My project has three tasks need to be done.
# First, to create a list to show all the txt files I will analyze in the following sections.
# Second, I what to figure out 10 major topics my data set includes.
# Third, to visualize my topic modeling results.

import os
import gensim, nltk
from nltk.corpus import stopwords
from stop_words import get_stop_words
import pandas as pd
import matplotlib.pyplot as plt
from os import walk
import numpy as ny
import csv

# The first task I want to do, is to create a list in which I can see the name of the files and the number of them.
# The code here is my first attempt.

# The following code direct us to data set which will be used in this project.
path = "/Users/hinmingfrankiechik/Desktop/text_files"

# I am trying to store all the file names in this list
filelist=[]
ignore_files={'.DS_Store', '.txt', 'desktop.ini', 'Huovinen.dataset'}
sorted(filelist)

for root, dirs, files in os.walk(path):
    for file in files:
        if file not in ignore_files:
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

# Before processing Topic Modeling, I will add a stop words list to my data set so that no common words in English will be analyzed in my code.
# This is a simple stop words list in NLTK.
stop_words = set(stopwords.words('english'))
print("Stop Words", list(stop_words)[:30])

# However, the list originally in NLTK is not enough. I have tried to analyze my data set but some words like 'Author', numbers of years were included.
# Thus, I have to create an extended stop words list.
new_sw = [
    'JSTOR', 'Author', 'which', 'who', 'hello', 'school', 'from', 'subject', 'edu', 'use',
    1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
    'pp', 'data', 'al', 'vol', 'Open', 'Access', 'Accepted', 'Received', 'article', 'available' ,'License'
    'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December',
    'grateful', 'total', 'member'
]
new_stopwords_list = stop_words.union(new_sw)
print(len(stop_words))

texts = []
labels = []

for path, dirs, files in os.walk('/Users/hinmingfrankiechik/Desktop/text_files'):
    for file in files:
        if file not in ignore_files:
            with open(os.path.join(path, file)) as rf:
                text = rf.read()
                tokens = nltk.word_tokenize(text)
                cleaned = [word for word in tokens if word.isalnum()]
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
    prefix='txt200_'
)

lda_model.save("CDH598_200.p")

topics = lda_model.show_topics(num_topics=number_of_topics, num_words=20)

for topic in topics:
    print(f"Topic #{topic[0]}: {topic[1]}...\n\n")

# Now, let's visualize the topic modelling result.
# I create a function that allows me to transfer the results of each year into pandas. After that, the results shown in pandads will be visualized in mutiple bar charts.

names = ['Topic', 'Weight', 'Words']
def read_table (txt_files):
    list(open(txt_files))
    df = pd.read_table(txt_files, names=names)
    del df['Words']
    print(df)
    df.plot.bar(x='Topic', y='Weight')
    plt.title("Topic Weights")
    plt.tight_layout()
    plt.show()
    return

df2 = read_table ('/Users/hinmingfrankiechik/Desktop/txt200_topickeys.txt')
print (df2)
