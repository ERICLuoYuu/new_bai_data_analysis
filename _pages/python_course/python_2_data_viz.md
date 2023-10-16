---
title: "Of plots and pandas: Data handling and visualization"
permalink: python_2_data_and_data_viusalization
author_profile: false
sidebar:
  nav: "python"
---

In this second part of the course we will talk about how to handle, process and visualize data in Python. For that purpose we will make use of a few third-party libraries. NumPy and Pandas will help us store the data in array- and matrix-like structures (in Pandas more specifically Dataframes) and do some processing of the data. Pandas already has some visualization capabilities, but for nicer looks and configurability we will make use of the Plotly package. Finally for some analysis of the data we will also take a quick look into the widely used Sciki-Learn package.  
To underline that these are essential tools in Python, let me once again pull out the Stackoverflow 2023 survey: According to the ~3k respondants, Numpy, Pandas and Scikit-Learn are 3 of the 8 most used technologies in programming across all languages (disregarding web-technologies)!  
![Stackoverflow 2023 survey technologies](/assets/images/python/2/technologies.PNG)  

For this part we will use some example data. It is a dataset from the german weather service DWD from the Diepholz Station (ID 963) ranging from 1996 to 2023. [Click here to download (31mb)...](/assets/data/diepholz_data_1996_2023.parquet).  
  
**Note** The data is in .parquet-format. You may not have heard of it, but this is a very compressed and fast format. For example this dataset with 27 years worth of data, in Parquet this is 31mb of data, in .csv its 208mb.  
While you can not open .parquet directly in excel or a text editor like a .csv file, it is much much faster to load e.g. when using it in programming languages, which is exactly what we are going to do here.
{:.notice}
  
As a last note: NumPy is one of the older Python libraries and Pandas is actually built on top of it. However, because we work with example data and want to get hands-on as fast as possible, we will cover Pandas first and then go from there.  


#### Table of Contents
1. [Pandas](#1-pandas)
2. [NumPy](#2-numpy)
3. [Plotly](#3-operators)
4. [Scikit-Learn](#4-loops-and-conditionals)

## 0. Importing modules
Just a quick forword on importing libraries in Python. Pandas, Plotly and Numpy are all external libraries we need to import to our script in order to make them work. Usually we would also have to install them, but since we work in Anaconda, this is already taken care of for us!  
Very simply, to import a library you type "import" and then the respective name. Typically you want to give an "alias" to the package, which is basically a variable that you can then use to access all the methods in the package. For some packages there are long-standing standards of what names to use. For pandas for example this is "pd":

```python
import pandas as pd
```

You can also only import specific parts of a package, which can save memory. Going back to one exercise from the previous lesson, if you know that you will only use the sqrt function from the math package you can use the syntax
```python
# Importing only a single function, squareroot
from math import sqrt

# Importing several functions, squareroot and greatest common divider
from math import sqrt, gcd

# theoretically you could also give an alias here
from math import sqrt as squareroot # this does not make much sense though
```

## 1. Pandas
Pandas is around since 2008 and one of the most wiedely used tools for data analysis of all. The usage is all about two types of objects: The pandas Series and the Pandas DataFrame, where a Series is more or less one column of a dataframe (basically a vector). If you already worked with R, the concept of a DataFrame is not new to you. However for starters, a DataFrame is basically a table, in which each row has an index and each column has a label. Simple right?  
  
![Pandas Dataframe Strcuture](/assets/images/python/2/pandas_df.png)  
(credit: https://www.geeksforgeeks.org/creating-a-pandas-dataframe/)
  

### Creating DataFrames
Lets create a first little DataFrame. There are several ways to do it, one rather intuitive way is to use a dictionary. Think about it, a dictionary already has values which are labeled by keys. You can easily imagine this in a table-format: The keys will be the column-labels and the indices (row-labels) are by default just numbered.  
```python
import pandas as pd

# Note that we create an instance of the class "DataFrame"
# Therefore we have to call the function pd.DataFrame(). Within
# the brackets we then define a dictionary using the {}-style syntax
values_column_1 = [2,4,6,8,10]
values_column_2 = [3,6,9,12,15]
df = pd.DataFrame({
    "column_1": values_column_1,
    "column_2": values_column_2
})
```  
  
Another option to create a dataframe is of course to read in data. Lets go ahead and read the data from the german weather service that you can download above. Now we can use pandas built-in data-reader to directly create a DataFrame from the parquet-file:  
```python
# The path can either be the absolute path to the place where you saved the file
# or the relative path, meaning the path relative to the place where your script is.
# I'd recommend to create a subfolder where your script is called "data" and then
# import the data from the path "./data/diepholz_data_1996_2023.parquet"
df_dwd = pd.read_parquet('path_to_file')
```
Once you createad a dataframe, you can access individual columns by using the column names. Either you can directly access them using brackets, or you use the built-in ".loc"-function. I would recommend getting used to the .loc right away, as it rules out some errors you can run into otherwise. With .loc you always have to provide first the rows you want to access and then the column, separated with a comma. If you want to get all rows, that is done using a colon (":") To get a list of all availabel columns you can simply type "df.columns"  
  
```python
# First we can take a look at the available columns
df_dwd_columns = df_dwd.columns
print(df_dwd_columns)

# Then we can use the column names to extract a column
# from the dataframe
# Either you use only the column name in brackets:
df_dwd["tair_2m_mean"]
# But even better: use the .loc function:
df_dwd.loc[:,"tair_2m_mean"]        # get all rows
df_dwd.loc[20:50,"tair_2m_mean"]    # get rows 20 to 50
df_dwd.loc[:20,"tair_2m_mean"]      # get all rows up to 20 (including 20)
df_dwd.loc[20:,"tair_2m_mean"]      # get all rows after 20 (including 20)
```
Note that the .loc examples above all assume numeric index. But Pandas is not restricted to that!
The index (or "row-label") could also be something like "mean" or "standard-deviation".  
Keep that in mind for the exercise below!
  
Pandas has a great set of convenience functions for us to look at and 
evaluate the data we have.  
- .info() gives us a summary of columns, number of non-null values and datatypes  
- .head() and .tail() show the first or last five rows of the dataframe  
- .describe() directly gives us some statistical measures (number of samples, mean, standard deviation, min, max and quantiles)  
  
Note that the output of .describe() is again a DataFrame, that you can save in a variable to evaluate it.  
  
{% capture exercise %}

<h3> Exercise </h3>
<p >You already know, how to call a method that is attached to a class. With that knowledge, explore the Diepholz DWD dataset and figure out the mean, standard deviation, min and max for 
air temperature, precipitation height, air pressure and short wave radiation (SWIN)</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Hint</summary>
It may be that the output of the .describe() function has a pretty bad formatting with 5 decimal numbers or more. In that case you can change the formatting of the output using  
```python 
df.describe().map('{:,.2f}'.format)
```
</details>

<details><summary markdown="span">Solution!</summary>

```python

# First get the summary buy saving the output of .describe()
# in a new dataframe
df_dwd_summary = df_dwd.describe().map('{:,.2f}'.format)

# Then to access a value of interest:
tair_2m_mean = df_dwd_summary.loc["mean", "tair_2m_mean"]
tair_2m_min = df_dwd_summary.loc["min", "tair_2m_mean"]
# and so on...
```
</details>

<h3>Challenge </h3>
There is a one-line solution to this task, that only grabs the values asked for in the exercise. I wouldn't say that that would be the recommended solution for the sake of overview, but to fiddle around it is a good challenge. Hint: You can pass lists for the row- and column-labels to .loc
<details><summary markdown="span">Solution</summary> 

```python
# We can chain all the commands above to a one-line operation, meaning we 
# directly call .describe().map().loc[] on each others output.

# By passing the list ["mean", "std", "min", "max"] as row-indices and 
# ["tair_2m_mean","precipitation","SWIN","pressure_air"] as column-labels 
# we can directly access the range of values asked for in the exercise.

df.describe().map('{:,.2f}'.format).loc[["mean", "std", "min", "max"], ["tair_2m_mean","precipitation","SWIN","pressure_air"]]
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
