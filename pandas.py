import pandas as pd
import matplotlib.pyplot as plt

names = ['Topic', 'Weight', 'Words']

list(open('txt2001_topickeys.txt'))
df = pd.read_table('txt2001_topickeys.txt', names=names)

del df['Words']
print(df)

df.plot.bar(x='Topic', y='Weight' )
plt.show()



