import pandas as pd

names = ['Topic', 'Weight', 'Words']

list(open('txt2001_topickeys.txt'))
result1 = pd.read_table('txt2001_topickeys.txt', names=names)
print(result1)

list(open('txt2008_topickeys.txt'))
result2 = pd.read_table('txt2008_topickeys.txt', names=names)
print(result2)



