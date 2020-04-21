import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import andrews_curves

names = ['Topic', 'Weight', 'Words']

list(open('txt2001_topickeys.txt'))
result1 = pd.read_table('txt2001_topickeys.txt', names=names)
print(result1)

list(open('txt2008_topickeys.txt'))
result2 = pd.read_table('txt2008_topickeys.txt', names=names)
print(result2)



