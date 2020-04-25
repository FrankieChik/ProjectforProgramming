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
