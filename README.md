# Programming for Humanities 
## Final Project

1. Introduction  
This is a project for CDH 598: Programming for Humanities.  
In this project, I attempt to practice my skill of writing python code.  
As will shown, the project has different sections.  
The corpus I will use **text_files** in this project is provided by the instructors.  
In preparation, I have read three articles provided. They are:  
  *  
  *  
  *  

2. **Bibliography**  
I am not familiar with the corpus. Thus, I have to know how many documents my corpus have and the number of them in each year.  
I create a bibliography and a bar plot in this part of my project.  
You can find the code in Bibliography.py
  * Bibliography  
  I first use the **os.walk()** in Pyhton to go over the entire corpus.  
  The documents are all listed in a list entitled **filelist**.  
  The problem I first encountered is that the files are not listed in chronological order. 
  Also, I see some documents which are not in txt.file format.  
  Thus, I add a list which contains all non txt.files that I should ignore in my analysis.  
  Also, I have to use **dirs.sort()** to sort all subdirectories so that all files will list chronologically.  
  My method in creating a bibliography is to use **pandas dataframe**. The titles of all documents are presented in a Series of this pandas dataframe. Since the titles of the documents consist of the authors' names, years, and titles, with each component separated by "_-_", I can split the Series into three columns by "_-_" accordingly.  
  My bibliography shows that there are 400 files in totoal. 
  * Bar plot  


3. **Preprocessing Data**  
<>
