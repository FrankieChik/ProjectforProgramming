import gensim, nltk, os


ignore_files = {'.DS_Store', '.txt'}

texts = []
labels = []

for path, dirs, files in os.walk('/Users/hinmingfrankiechik/Desktop/text_files/2008'):
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
