# Programming for Humanities: Final Project  
## Introduction  
This is a project for CDH 598: Programming for Humanities.  
In this project, I attempt to practice my skill of writing python code.  
As will shown, the project has different sections.  
The corpus I will use **text_files** in this project is provided by the instructors.  
This corpus has **nine** subdirectories, and all files are categorized into these nine subdirectories according to the years they were published (2001-2009).
In preparation, I have read three articles provided. They are:  
  *  
  *  
  *  

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
  
## Preprocessing Data  

<>
