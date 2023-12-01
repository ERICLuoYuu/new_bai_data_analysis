---
title: "4. Extreme value detection"
permalink: /python_4_extreme_detection/
author_profile: false
sidebar:
  nav: "python"
---

In this exercise we will look at extreme values in meteorological data. First you will learn about different ways to define what is an "extreme" 
value (extreme relative to what?). Afterwards we will work with an example dataset and code the methods in Python to get some hands on experience.

#### Table of Contents
1. [Material](#1-material)  
2. [Background](#2-background)  
3. [Methods and Implementation](#3-methods-and-implementation)  
  3.1. [Peak Over Threshold (POT)](#31-peak-over-threshold-pot)  
  3.2. [Block Maxima (BM)](#32-block-maxima-method-bm)  
  3.3. [Moving Average (MA)](#33-moving-average-method-ma)  


### 1. Material
We will once again use the DWD dataset from the Diepholz station for this section. Below you can find the download again:
[Diepholz DWD meteo data (25mb)...](assets/data/dwd_diepholz_1996_2023.parquet).  
[Literatur](/assets/r_ex4/wmo-td_1500_en.pdf)  

We will have to do some plotting again, so it might be good for you to resample the data to hourly data again, just to reduce the size
of the dataset a bit. By now you should know how to do it. Try to create a pandas Series with hourly air temperature data.  
If you are getting stuck, you can refer to the exercise before.
```python
df_dwd_ta_hourly = ...
```


### 2. Background
We will look at three different methods to determine extreme events from time series of meteorological data. The main difference between the methods is the way they define the reference, to which we compare a value to describe it as being "extreme" or not.  
Pause for a second and think about how you could describe what an extreme value is.  
  
There are several ways to think about extreme values. An extreme value can simply be the highest/lowest value in a finite set of data. Think for example about testing the highest speed that a car reaches on a test drive. Here the absolute peak value would be a reasonable value of interest.  
  
In meteorological time series we are often interested in a range of extreme values. In other words, we are interested in the values in the tails of the distribution of our sample data, which exceed a certain threshold. It is important to consider the distribution of our underlying dataset and the question we actually want to answer. 

Our example dataset comprises of air temperature data from 1996 to 2023. If we are interested in the extreme values with respect to this whole time period, we can simply look at the distribution of all the data, determine a threshold and see which datapoints are above the upper or below the lower threshold. 

However, we might also be interested in the months with extreme temperatures. Because our distribution includes winter and summer data, extreme temperatures in spring and autumn will probably not be considered in this approach. For these we would have to create data distributions of seasonal, monthly or even daily data to evaluate extreme events on the respective time scale. This will become more obvious when we look at the methods.

<details>
<summary>
Read More: Extreme value return periods
</summary>
Another approach is the evaluation of extreme values and their probabilities based on historical data. Relating these probabilities to the time series of the data produces "return periods", frequencies in which the extreme values are expected to occur. As an example, requirements for buildings often include a resistance to weather extremes with a certain return period. Making up a case, wind turbines would be built that they can withstand windspeeds with a "return level" in a "return period" of 1 in 10.000, meaning the chance that such a windspeed occurs in a year would be 0.01%.
</details>

### 3. Methods and Implementation
Lets now look at three different methods to analyze extreme events in our sample dataset. We will talk about the reasoning and the implementations of the methods. 
We will then go through each method and implement the methods into functions, which you can then use to analyze your data.

We will start by defining a so called "wrapper-function". This will be a function we can pass our data to and define the method, we want to use for extreme detection.
The function will then call another "sub-function" for us, which actually does the extreme detection. So the arguments that this wrapper function needs are the actual data, a quantile argument and and an argument defining the method we want to use.

What are quantiles?  
  
A quantile is a subset of the given data, that contains a certain percentage of the distribution of the data. E.g. in the following figure, everything left of the red line is in the q10 percentile, the lowest 10% of the data. Everything up to the blue line is in the q90, the quantity that comprises of 90% of the data.
![Quantiles](.\misc\2023\01\05\quantiles.png)
You can use the following function to visualize the quantiles of our dataset:
```python

def visualize_quantiles(x:pd.Series, q_low:float, q_high:float):
    import scipy.stats as stats
    x_mean = x.mean()
    x_sd = x.std()
    y = stats.norm.pdf(x.sort_values(), x_mean, x_sd) # This function creates the y-values of the normal distribution given our data, the mean and the standard deviation
    qh = x.quantile(q_high) # here we calculate the higher quantile threshold
    ql = x.quantile(q_low) # here we calculate the lower quantile threshold 
    fig = px.scatter(x=x.sort_values(),y=y)
    fig.add_trace(
        go.Scatter(
            x=[ql, ql],
            y = [0,max(y)],
            mode="lines",
            name=f"{q_low*100}% quantile"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[qh, qh],
            y = [0,max(y)],
            mode="lines",
            name=f"{q_high*100}% quantile"
        )
    )
    fig.show()

get_quantiles(df_dwd_ta_hourly, 0.05, 0.95)
```
Back to our extreme_detection function:  

The return type will be a pandas dataframe. Overall the function definition will look somewhat like this:
```python
def extreme_detection(X:pd.Series,quantile:float,method:str) -> pd.DataFrame:
  IMPLEMENTED_METHODS = ["POT", "BM", "MA"]
  if method not in IMPLEMENTED_METHODS:
    raise Exception(f"Method {method} not implement. Use one of {IMPLEMENTED_METHODS}")
  if method = "...":
    ...
  elif method = "...":
    ...
  elif method = "...":
    ...
  return ...
```
The arguments for this functions are the following:
- **X** is the series of data which is to analyzed for extremes
- **quantile** defines the extreme threshold in terms of percentiles of the data given as floats, e.g. 0.8, 0.9. E.g. passing in 90 will define everything above a range of values that includes 90% and below a range that includes the lowest 10% as extreme high/low values.
- **method** defines which extreme detection will be used
  - "POT" for Point Over Threshold
  - "BM" for Block Maxima 
  - "MA" for Moving Average.

The structure of the method is pretty simple:  
First we check whether the method that as given is actually an implemented one. By convention, variable names for constants are written in all-capslock. So we have a list of implemented methods, and if the provided method is not in that list, we throw an error at the user that tells him/her, that this is not a valid input. 
Depending on the given method argument, one of three blocks of code will be executed. The branching is simply done with if-conditionals.

#### 3.1 Peak Over Threshold (POT)

The first approach is the Point Over Threshold (POT) method. This is a very simple approach that looks at the whole dataset as one.  
We define fixed thresholds for the dataset, defining the upper and lower bounds above or below which values will be considered extreme.  
The boundaries are usually defined by the quantiles we provided as an argument to the function.

{% capture exercise %}

<h3> Exercise </h3>
<p >Lets try and code that method ourselves. It is actually not very difficult! <br>
Define a new function called "peak_over_threshold()". <br> 
It needs to take a series of data as input and the quantile we want to use for extreme detection. <br>
Then we need to do the following operations: <br>
<ol>
  <li>Find the upper and lower thresholds for what is to be defined as extreme, absed on the quantiles. To find these values you can use the handy Python function "quantiles()". Just call it on the input Series and provide the quantiles as argument as in "X.quantiles(0.95)". Remember: You want the upper **and lower** thresholds. Think about how you can get both.
Then create a dataframe to return with two new columns: One called
  <li> Find those rows in the input series which are higher and lower than the upper and lower thresholds. You can get Series of booleans by comparing a pandas Series with a value. You can try it out, just type for example "X > 270" if X is your Series.
  <li> Finally you want to create a dataframe, because of course you want to return the results of your extreme value detection. Create a dataframe with the input data and two new columns, one containing the booleans of your high extreme values and the other for the low extremes.
</ol>
A little hint: The description here is quite long but the code for this is actually quite short.

{::options parse_block_html="true" /}

<details><summary markdown="span">Hint if you get stuck!</summary>
You can generate a Series of boolean values that indicate whether a datapoint is above or below a value with a direct comparison such as 
```python
    X_larger_than_280 = X > 280
```
</details>


<details><summary markdown="span">Solution!</summary>

```python

def peak_over_threshold(X:pd.Series, prob):

    print(f'Extremes detection using peak over threshold method at: {prob} percentile')
    df = pd.DataFrame(index=X.index, data = {
        "data": X,
        "extreme_low":  X < X.quantile(q=1-prob),
        "extreme_high":  X > X.quantile(q=prob)
    })
    return df
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>


Very simple: First we create a new Dataframe with our data and the respective dates.  This dataframe is fed into a pipeline where we "mutate" the dataframe. In the mutation, a new column is added called "Extreme". This column is then populated through a conditional function, the if_else() function from the dplyr package. Here the code gets a bit complicated, because there is a second if_else() inside the first if_else() function. It is a "nested" function.  It works like this:  
Normally the if_else() function checks a condition and for all values passed and returns one of two values, depending on whether the condition is true or false. Consider this:

In our POT-codeblock we first examine if the datapoint is bigger than the upper quantile threshold. If so, it sets the value in the "Extrem" column to "Extreme-high". If not, the second if_else block checks whether the value is lower thant the lower quantile boundary. If so, it sets the value of the "Extreme" column to "Extreme-low", else to "Not-Extreme".

Go ahead and run the example for the POT method:
```R
## Extreme at 95th percentile using POT method---
Tair_extreme_POT = Extreme_detection(TairData$Tair_f, TairData$Date, 95, 'POT')
```
You will get a plot of the data with marked extreme low and high values (see spoiler below) as well as a dataframe that contains the "Extreme" column along with the original data.

Output Figure:  
![POT image](/assets/r_ex4/extreme_POT.png)
*Figure 1: Extreme values based on the whole dataset POT method*
  
As expected you can see that this plot gives us information about the extreme values with respect to the whole timeline. 

---------
#### Exercise

1. Lets fiddle with the code for a bit. Change the **prob** parameter to 85, 75 and see how the output changes. How many extreme values do you expect when setting prob to 50? Think about it and then run the function with that quantile.
2. In the additional methods at the end of the script you can find a function called ``` Extremes_per_year()```. Run it with your output from the POT extreme detection with different quantiles as argument and evaluate the output. What do the trend lines indicate? 
Extra: If you want you can evaluate the trend lines by fiddeling with the function code. Remember the lesson about linear models and the "summary()" method. How would you describe the usefulness of the linear model fitted to this data?
3. You can visualize the thresholds that define extreme values using the ```Quantiles()``` function which is given in the additional functions. Run it with the output of your extreme detection function for a few different prob arguments. Note that you have to pass the lower and upper quantiles to the ```Quantile()``` function, e.g. for 95 % you call it as ```Quantiles(dataframe, 5, 95)```. Look at the different plots, what do you think would be an appropriate threshold?

----------

#### 3.2. Block Maxima Method (BM)
Block maxima codeblock:  
```R
  if(method == 'BM')
  {
    sprintf('Extremes detection using block maxima at %f percentile',prob)
    DF = data.frame(Value = X, Date = timecol) %>% mutate(DOY = as.numeric(format(Date,'%j'))) # calculating DOY for long-term mean, calculate month if working on monthly data.
    
    DF = DF %>% mutate(Var_15ma = lead(c(rep(NA, 15 - 1),zoo::rollmean(Value,15, align = 'center')),7)) %>%   ## 15-day moving average and then long-term mean
      group_by(DOY) %>% 
      mutate(Var_LTm = mean(boot(Var_15ma,meanFunc,100)$t, na.rm = TRUE))  %>%   ## Var_LTm is the long-term mean based on a 15-day moving average-
      mutate(del_Var = Value - Var_LTm) %>%       ## deviation from a long-term mean
      mutate(Extreme = if_else(del_Var > quantile(del_Var, probs = prob*0.01, na.rm = TRUE), 'Extreme-high',
                               if_else(del_Var < quantile(del_Var, probs = (1-prob*0.01), na.rm = TRUE), 'Extreme-low', 'Not-Extreme')))  ## Assigning extreme classes based on del_Var
    
    p = DF %>% 
      ggplot(., aes(x = Date)) + 
      geom_point(aes(y = Value, fill = Extreme), color = 'black', stroke = 0, shape = 21) + 
      geom_line(aes(y = Value, color = 'Value'), size = 0.4) + 
      theme_bw() +
      geom_line(aes(y = Var_LTm, color = 'Long-term mean'), size = 0.8) + 
      scale_color_manual('', values = c('black','grey50'))
    
    print(p)
    
    return(DF)
  }
  ```  

The next method we are looking at is the "Block Maxima" method. As the name states, we are looking at a certain "block" of data and find the maxima based on the defined threshold of the values in this block. There are several ways we could define these reference blocks. For example we could look at every year individually and find the extreme values for these. We would get an array of the hottest and coldest days of each year separately.
If we where more interested in extreme values across years, we could for example define a block as data from each season across the years. So the block "spring" would consist of data from 01.03. to 31.05. across all the years in the dataset. We could then find extremes based on the quantiles of the seasonally data and separate e.g. extreme values in spring and autumn from the overshadowing extreme values in winter and summer.

In our example code the blocks are defined as the values for each single day across all the years. The procedure is as follows: 

**Step 1**  
In the first line we again create a dataframe with the data, a date column and then add a new column called "DOY" with a mutation, that gives each date a value of 1 to 365. We can later use these values to calculate a mean for each year across the seasons:  
```R
 DF = data.frame(Value = X, Date = timecol) %>% mutate(DOY = as.numeric(format(Date,'%j'))) # calculating DOY 
 ```

**Step 2**  
Next a second mutation is being done, which creates the column "Var_15ma". This is the 15 day moving average around every day. Moving average means that the "window" of data we are calculating the mean from varies. For each data point  
```R
 DF = DF %>% mutate(Var_15ma = lead(c(rep(NA, 15 - 1),zoo::rollmean(Value,15, align = 'center')),7)) # 15-day moving average and then long-term mean 
 ```  
Through this, we accquire a smoothing of the daily temperature values and make the underlying dataset for our daily temperature distribution more broad. The reasoning is the following:  
We want to create a representative dataset for daily temperature values across the years. If we use the single day for each year, we have a dataset of 18 datapoints which can easily include heavy outliers. By using a moving average of 15 days we enhance our dataset for each day by a factor of 15 to 270 datapoints, still restricted to a pretty small time window. While it does reduce the impact of individual extremely hot or cold days, it is more likely to representatively capture the state of the atmosphere around the time of interest.  
  
**Step 3**  
The next step creates the column "Var_LTm". This is the column representing the long-term mean (LTm) for each day of the year:  
```R 
mutate(Var_LTm = mean(boot(Var_15ma,meanFunc,100)$t, na.rm = TRUE))  %>%   ## Var_LTm is the long-term mean based on a 15-day moving average-
```  
This is a bit of a nested function call because we use a method called "bootstrapping". Bootstrapping means that we take several random subsamples from the data we have and calculate the mean for each subsample. Then we can look at the distribution of these means e.g. to find confidence intervals. This allows us to asses how representative our mean value for the whole dateset is. Finally, we can calcualte the mean of the means of the subsamples and use that as our final mean value. The function looks like this:  
```R
mean(boot(Var_15ma,meanFunc,100)$t, na.rm = TRUE))
```  
The order in which these functions are executed is form the inside out: first the function ```boot()``` is called with the parameters Var_15ma, meanFunc and 100. This means we take 100 subsamples from the column Var_15ma and call the function ```meanFunc()``` on them. This is just a function that calculates the mean of a vector. With the $t after the function call we access the resulting vector of 100 means which is returend from the boot() function. Then we simply calculate the mean from these means and ingore the NA-values with ``` mean(..., na.rm = TRUE)```.  
  
**Step 4**  
Finally we calculate the deviation of each datapoint from the previously derived mean and create a new column called "del_Var". Now these deviations from the mean are split into quantiles and del_vars which are above or below the threshold are defined as extreme:  
  
```R
mutate(del_Var = Value - Var_LTm) %>%       ## deviation from a long-term mean
mutate(Extreme = if_else(del_Var > quantile(del_Var, probs = prob*0.01, na.rm = TRUE), 'Extreme-high',
                          if_else(del_Var < quantile(del_Var, probs = (1-prob*0.01), na.rm = TRUE), 'Extreme-low', 'Not-Extreme')))  ## Assigning extreme classes based on del_Var
    
```  
Note: In the POT approach the quantiles where built from the whole dataset itself. Here, the quantiles are built from the array of deviations from the mean! Remember this in the exercise when you evaluate the results.

---
### Exercises

1. After using POT and the BM method with daily blocks, which method do you expect to yield more extreme values per year? Think about it and then use the ```Extremes_per_year()``` function to check your hypothesis. You can also print a table of the number of high and low extremes using the given function ```Total_Extremes()``` which also takes a dataframe as an argument.
2. The Extreme_Detection() function has one parameter called "rollmean_period" which has a default value of 15. This means that the time window on which the mean for each day is calculated is 15 days, the day + the 7 days before and after. Think about how it might affect the outcomes if you set this value to 1 or to 365. To evaluate, you can use the extra functions given in the end of the script, ```Plot_Extremes()``` and ```Quantile()```. Remember what the quantiles where built from (so which column you have to pass to the ```Quantile()``` function). You can use the ```Plot_Extremes()``` function to plot the extreme values of a specific year e.g. for 2017 like this:  
```R
Plot_Extremes(Tair_extreme_MA %>% filter(Date > '2017-01-01' & Date < '2018-01-01'))
```

---

#### 3.3. Moving Average Method (MA)
Moving average codeblock:  
```R
  if(method == 'MA')
  {
    sprintf('Extremes detection using "moving average" at %f percentile',prob)
    DF = data.frame(Value = X, Date = timecol) %>% mutate(DOY = as.numeric(format(Date,'%j'))) # calculating DOY for the long-term mean, calculate month if working on monthly data.
    
    DF = DF %>% 
      arrange(Date) %>%  
      mutate(Var_15ma = lead(c(rep(NA, 15 - 1),zoo::rollmean(Value,15, align = 'center')),7)) %>%   ## 15-day moving average 
      mutate(del_Var = Value - Var_15ma) %>%       ## deviation from a 15-day moving average (change the days as per requirement)
      mutate(Extreme = if_else(del_Var > quantile(del_Var, probs = prob*0.01, na.rm = TRUE), 'Extreme-high',
                               if_else(del_Var < quantile(del_Var, probs = (1-prob*0.01), na.rm = TRUE), 'Extreme-low', 'Not-Extreme')))  ## Assigning extreme classes based on del_Var
    
    p = DF %>% 
      ggplot(., aes(x = Date)) + 
      geom_point(aes(y = Value, fill = Extreme), color = 'black', stroke = 0, shape = 21) + 
      geom_line(aes(y = Value, color = 'Value'), size = 0.4) + 
      theme_bw() +
      geom_line(aes(y = Var_15ma, color = 'moving mean'), size = 0.8) + 
      scale_color_manual('', values = c('black','grey50'))
    
    print(p)
    
    return(DF)
  }
  ```
  
The final method we will look at is the moving average method. As the name already states, here the extremes are detected on a more temporally constrained basis, the moving average around each datapoint. Take a look at the code block above for the moving average method. Everything used here was already used in the blocks before, only the deviation from the mean (the "del_var") is now computed differently.  

----

### Exercises

1. Think about how using a smaller time reference window might affect the extreme value detection. Would you expect extreme values in this approach to be more or less frequent than in the block averaging method? Then run the detection function and save the output in a new variable. Finally use the given evaluation function ```Total_Extremes()``` and pass it the output. Was your guess right?  
2. You have now run all three methods. In the evaluation functions you have a given function ```Plot_All_Extremes()```. Call it with your POT, BA and MA outputs as arguments and look at the temperature ranges which where categorized as extreme values. Write up a very short summarization.   
3. Change the parameter rollmean_period of the extreme detection function with the MA method to 365 and pass that output to the ```Plot_All_Extremes()``` function. How do you explain the output in comparison to the other methods?  

----
