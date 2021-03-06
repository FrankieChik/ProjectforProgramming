# Programming for Humanities: Final Project  
## Introduction  
This is a project for CDH 598: Programming for Humanities.  
In this project, I attempt to practice my skill of writing python code.  
As will shown, the project has different sections.  
The corpus I will use **text_files** in this project is provided by the instructors.  
This corpus has **nine** subdirectories, and all files are categorized into these nine subdirectories according to the years they were published (2001-2009).
In preparation, I have read two articles provided. They are:  
  *  "Topic modeling made just simple enough"  
  *  "Analyzing Documents with TF-IDF"  
The entire code is in 05042020.py.  

## Bibliography  
I am not familiar with the corpus. Thus, I have to know how many documents my corpus have and the number of them in each year.  
I create a bibliography and a bar plot in this part of my project.  
You can find the code in Bibliography.py
1. Bibliography  
  I first use the **os.walk()** in Pyhton to go over the entire corpus.  
  The documents are all listed in a list entitled **filelist**.  
  The problem I first encountered is that the files are not listed in chronological order. 
  Also, I see some documents which are not in txt.file format.  
  Thus, I add a list which contains all non txt.files that I should ignore in my analysis.  
  Also, I have to use **dirs.sort()** to sort all subdirectories so that all files will list chronologically.  
  My method in creating a bibliography is to use **pandas dataframe**. The titles of all documents are presented in a Series of this pandas dataframe. Since the titles of the documents consist of the authors' names, years, and titles, with each component separated by **"__-__"**, I can split the Series into three columns by "__-__" accordingly.  
  My bibliography shows that there are 400 files in total (see Bibliography.csv).  
2. Bar plot  
  After having a bibliography, a create another pandas dataframe to calculate the total number of publications in a year.  
  Based on this pandas dataframe, I have drawn a bar plot to show the number of documents in each subdirectory (see Number of Publications.png).  
  We can see on the bar plot that most of the files in my corpus are from the subfirectory of 2008 (more than 175), with the 2009 immedately follows it (around 150).  
  In contrast, the numbers of publications in the first six years are drastically lower than the numbers in 2008-2009.  For instance, 2001 only contains one documents.  
  
## Topic Modelling 
The bibliography I created also tells me that many of the articles are related to ecology and biology. However, what particular topics this corpus contains?  
In order to figure out the hidden topics in my corpus, I would first need to clean the texts.  
You can find the code of this part in **topicmodel.py**.  
1. Cleaning the texts  
  The first step of cleaning the texts is to implement a stopword list. The 'english' stopwords list originally from nltk contains only 175 English common words.  
  Therefore, I used **FreqDist()** to sort out the 50 most common words in my corpus after implementing the original stopwords list. 
  To do this, I should first tokenize the files and delete all unnecessary tokens, like punctations, numbers. You can see in the **Proprocessing.py** the series of steps I walked through to clean my corpus.  
  After tokenizing the files, I used **isalnum(**) and **isnumeric()** to delete the punctations, numbers. Moreover, **stem()** and **lemma()** were used to turn every word in its original form. All words less than 2 characters are also deleted by the code of **len(word) > 2**.  
  **lower()** is also an important step that allows me to convert all words into lowercase. Thus, the words in my expended stopwords list can be typed in lowercase form.  
  **bigram** and **trigram** are also used, but I am not 100% sure how effective they are in my preprocessing.  
  Through these processes, we come with a list of 400 lists of strings. I thereofre have **isinstance()** to convert them into a lost of strings without split the strings.  
  At the end, I can apply **FreqDist()** to search for 500 most common words in my corpus. 
  Here, I did not use the method in Lavin's article. My main reason is that my purpose is to know the most common word, but not to know the term frequency weights of each word. As a result, **FreqDist()** is enough to fulfill this purpose.     
  After a scrutiny of the most common words, I add those common but unnecessary terms like 'read', 'jstor', and 
  author' to my entended stopwords list **stop_words.extend** .  
  A new text **new_corpus** is now created for us to do a more accruate topic modelling. 

2. Saving the data  
  Unlike "Topic modeling made just simple enough", I decide to use LDA Mallet to conduct my topic modelling analysis.  
  My reason is that Mallet provides a more accruate reason. I have tried noth LDA Model and LDA Mallet. Although the time I spent on using the first one is less than that on the latter, the result of Mallet is suggested to be more accruate. Moreover, the Mallet allows me to save the topic modelling results in several files which can make my next step much easiler.  
  Before applying LDA Mallet, I have used **id2word()** to make a dictionary in which a word will be shwon in this way, for instance, {1, 2}. The key is the id of the word while the value is its frequency in a document.  
  **id2word.filter_extremes** allows me, on the other hand, to delete those words appear in less than 5 files (no_below=5). Should I also add (no_above=0.5)?   
  I can process my topic modelling. In my analysis, I decide to come up with *10* topics with each topic consists of *20* words.  The number of topics as well as the numbers of keywords are for my convience only. Of course, having *20* topics can make the project more complicated but accurated. But at this moment, I think even an analysis with *10* topics is enough to see any hidden topics in mu corpus.  
  My analysis was saved in multiple files. I can load *'CDH598_200.p'* by using **LdaModel.load** to get the result back. This file saves the sum of the topics and the key terms that form a certain topic with their term weights. I will use this file later.  
  *txt200_doctopics.txt* and *txt200_keytopics.txt* are both text files in which I can use to calculae the weights of each topic in the entire corpus and in a certain document.  
  I first see the weights of those topics in certain documents. Since the file contains some exponiental numbers, I used **float_format** to float the number into **5** decimal places. Indeed, before using **float_format**, I was wrong in calculating the dominant topic in each document. After carefully reading the csv.file, I find some values like 5.XXXXXXE-5 very strange. Becasue the panadas dataframe will not autiomatically change the exponiental numbers into decimal numbers, I should do it by myself. This means a careful review on the result is curcial to my interpretation of the corpus.  
  In order to know which is the dominant topic of a particular document, I use **np.argmax** to calculate the largest values on a row and show it in a new column named 'Dominant Topic'. We can conclude based on this information that in how many docunments a topic serves as the dominant topic.  
  The information in *txt200_keytopics.txt* is quite easy to understand. This document tells is the overall weights of each topic in the corpus. We can see in this document that topic number 8, which is the nineth topic, is the most dominant one.  
  
## Visualization  
1. In the first section in which I created a list of the data, I have already conducted my visualization. The figure **Number of Publications.png** is a line plot to show the number of publications by years.
2. The second visualization in my project is **Topic Weights in the Corpus.png**. This bar plot shows us the weights of each topic we have got from previous analysis. In order to create this bar plot, I have to split the data in **txt200_topickeys.txt'** into three groups by **\t** in a dictionary. After that, I could create a plot by correctly locating the data in the dictionary.  
3. I have also created bar plot **Dominant Topics in Documents.png**. Both bar plots tell me a very interesting thing. Although *Topic 9* is the most dominent topic in the entire corpus, it is Topic 7 in **Dominant Topics in Documents.png**, which is Topic 8 in **Topic Weights in the Corpus.png** serves as the dominent topic in most of the individual documents in the entire corpus. Does this mean that Topic 9 is fundamentally important in the corpus as a whole, even some documents do not treat it very well? Put in another way, Topic 9 may be a topic scholars have been concerned about. However, this may not be the main theme of every document. Nevertheless, it is relevant to the main theme of a specific document and a specific topic besides Topic 9.
4. You can find the code of my visualization in **visualization.py**.  

## Interpretation  
Based on *txt200_keytopics.txt*, we can determine the meanings of each topic. I interprete the meanings as follows (see *txt200_keytopics.txt* for the keyterms of each topic):  
0. This topic includes something relate to nutrition. Since rat and mouse (mous) are listed, it may talk about controlling the diet of a rat based on the nutrition of foods.  
1. This topic is related to the microbiota in a human's gut.  
2. This topic may suggest that human may be infected by mouse which contains pathogen.  
3. This is a topic of experiment in which a one's guts such as liver and gall are sample to be studied.  
4. This topic may talk about some research on the probability a baby or a child has of being infected due to the probiotic in his intestine.  
5.	This topic may touch on the issue of organ-cloning and organ transplantation becasue "clone" and "patient" are mentioned.  
6.	Related to animals, this topic cover the experiment different animals' propagation in different situations.  
7. This topic introduces a database in which one can find infomration about aniimals' genome or biology. It also covers instrcution on using this database.  
8.	It may be about the development of human's disease, and by touching an insect which contains pathogen, one would get sick.
9. This is about intestinal diseases and how one can prevent them.  

All topics are biological, ecological, and medical in nature. We can speculate, according to these ten topics, the entire corpus contains research paper which a scholars of biology, microbiology, nutriology, and pathology  will use.  

## Reflection  
I thiink a librarian may find the first section of my code useful in sorting out the materials he/she has. A researcher who is about to start his/her research project will also find it useful to get a firs impressioin on the corpus he/ she has (in terms of the size of the corpus).  
This is a humanist's preliminary attempt to write Python code to run a topic modelling. You may see that the entire code is only about 220 lines. I know some programmers can wrtie a code with more than 1000 lines. Thus, this code would not be too good enough. However, for those beginners, like me, who have no programming experience may use my code as an exampe to try a very preliminary and basic topic modelling analysis.  
Nevertheless, I find the section of preprocessing the data very useful and I have learned a lot from that. At the very beginning, my preprocessing is very simple and it took the computer so long to run the entire corpus (an hour?) However, after performing functions like **isnumeric()** and **id2word.filter_extremes**, my corpus is easiler to be analyzed and it only takes 2 minutes for my laptop to run the code and get the result. Although my preprocessing still has room to improve, I at least know that how to clean my data in a more detailed way.  Overall, the preprocessing code is very useful for my project as well as my future research.  
Meanwhile, visualization of the data shortens the time I have spent on studying the corpus. Although the pandas dataframe has already presented the result I need, it is time-consuming in go over the entire dataframe. The bar plots I created based on theose dataframes, in contrast, tell me the crucial information I need to know immediately. For instance, I know the ninth topic is the doinment topic in the entire corpus by reading the bar plot. *txt200_keytopics.txt* tells me its weight is **0.19995**, but I have to spend extra time on reading the weights of other topics, which I think is not so effective. Thus, I see visualization an essential step in presenting the result of my project and is more helpful than I expected.


