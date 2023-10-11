---
title: "Introduction"
permalink: python_1_basics
sidebar:
  nav: "python"
---

This interactive tutorial will get you started on your data-exploration road and make you familiar with some core
concepts of Python programming and data analysis.

#### Table of Contents
1. [General stuff about Python](#1-general-stuff-about-python)
2. [Data Types and Variables](#2-data-types-and-variables)
3. [Operators](#3-operators)
4. [Functions](#4-functions)
5. [Lists, Dicts, Numpy and Pandas](#5-data-structures)
6. [Loops and Conditionals](#6-extra-loops-and-conditionals)

Notice that it is not at all expected that you learn all these things and they are burnt into your brain (!!!!!). 
It is more of a broad intrdocution to all the basics so you have hard of them, but programmers do look up stuff
all the time! So don't worry if it is a lot of input right now, just try to understand the concepts and you 
can always come back and find help in here, in the internet or from me directly.
  
Here are some useful ressources if you get stuck:

[Link: Tutorials on many topics where you can quickly look up things...](https://www.w3schools.com/python/python_intro.asp)
[Link: Another nice overview of many functionalities of Python (requires login)...](https://www.w3schhttps://www.geeksforgeeks.org/python-cheat-sheet/)
Geeksforgeeks requires you to make an account or use e.g. a google login, but it features many tutorials, project ideas, quizzes and so on on many programming languages and general
topics such as Machine Learning, Data Visualization, Data Science, Web Development and many more
[Link: Later on we will use the library "Pandas" (so cute!) for data handling. A nice cheat sheet is provided by the developers...](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)


## 1. General things about Python
I quickly want to highlight some things which are special about Python compared to many other programming languages and that one needs to get used to.  

### Python is indentation sensitive
In Python it matters, how far you indent your lines, meaning how much space you have at the beginning of a line.
As an example this will work:
```python
a = 5
b = 1
```
but this will throw an error:
```python
a = 5
  b = 5
```
will result in an error:
```python
File "<stdin>", line 1
  b = 5
IndentationError: unexpected indent
```

### Variables
Generally in Python variables are created by assigning a value to them with an equal sign, just like we did above. Theire output can be shown by just typing the variable:
```python
a = 5
a
5
``` 

### Comments
Comments are lines in the code that are not executed and are there for documentation. For now it is a good idea to use comments in your code to keep track of what is happening where.  
Single line comments are always created with an '#'. Everything after that symbol in the line is not executed. Multi-line comments can be written by enclosing them in three ':
```python
# first I create a single line comment, this is not executed
a = 5 # this line is executed, but the comment gets ignored
''' 
Now I write a multi-line comment
I can continue the comment on the next line
b = 5 <-- this is ignored
'''

```


## 2. Data Types and Variables
As in all programming languages, there are different types of data in Python. 