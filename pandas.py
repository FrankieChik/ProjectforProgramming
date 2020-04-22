import pandas as pd
import matplotlib.pyplot as plt

names = ['Topic', 'Weight', 'Words']

# I create a function that allows me to transfer the results of each year into pandas. After that, the results shown in pandads will be visualized in mutiple bar charts.
def read_table (txt_files):
    list(open(txt_files))
    df = pd.read_table(txt_files, names=names)
    del df['Words']
    print(df)
    df.plot.bar(x='Topic', y='Weight')
    plt.show()
    return

df1 = read_table ('txt2001_topickeys.txt')
print (df1)

df2= read_table ('txt2008_topickeys.txt')
print(df2)
