---
title: "3. Interpolation and Gap Filling"
permalink: /python_3_interp_gapfill/
author_profile: false
sidebar:
  nav: "python"
---

In this exercise we look at interpolation and the rather time-series specific topic of data-gap filling.  
In time-series data we often have gaps due to a variety of reasons. They can result from instrumental issues or maintenance times or unfavorable weather conditions which leads to data being discarded. These data gaps can be filled with statistical methods.

In this lesson exercises are not completely separated from the content. Just follow along, grab the code and in some parts you will get snippets to run and fiddle with yourself.

Before we start plotting data we will see, how we can deal with missing values which are already handled by the institution
measuring the data, e.g. the DWD. For example it is common that the data is included with a specific placeholder value, which
we first need to handle.

### 1. Loading and  converting data:
We will use some data I have prepared in a way that you might find it in an online data portal.  
[Download the file here](/assets/data/dwd_diepholz_1996_2023_missing_placeholders.parquet)

To test some things we will work with the air temperature column "tair_2m_mean" here.
There are several issues when we have a missing-data-placeholder like that. Try two things:
 
{% capture exercise %}

<h3> Exercise </h3>
<p >Look at a quick express plot of the data. Is that a meaningful representation?
Then try to resample this data to daily values. Plot the data, do the values make sense?</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
df_dwd = pd.read_parquet(<path to file>)
df_dwd["date_time"] = pd.to_datetime(df_dwd["date_time"])

fig = px.scatter(df_dwd, y = "ta_2m_mean")
fig.show()
```
Plotting the data with the missing value placeholder makes the data barely readable:  
![Image of data with missing values](/assets/images/python/3/missing_values.png)

```python 
df_dwd_daily = df_dwd.resample(rule="d", on="date_time").mean()

fig = px.scatter(df_dwd_daily, y = "tair_2m_mean")
fig.show()
```
When averaging the values, the -999.99 values are taken into account leading
to unrealistic results:  
![Image of poorly average data](/assets/images/python/3/bad_averaged_data.png)

</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>


Lets look at the first way to solve this issue. We have to find the rows, where
the values are the placeholder value. You can identify these rows
by grabbing the right-row indices of the dataframe with a condition:  
```python
# we find the indices in the column of air temperature, where
# the values are -999.99:

#                                     |  here we find the rows we need:    | column we need|
#                                     | those where the row tair is -999.99|
#                                                        |                      |
#                                                        v                      v
indices_of_missing_values = df_dwd.loc[df_dwd["tair_2m_mean"] == -999.99, "tair_2m_mean"].index
```

We can then go ahead and replace these
values with a special type, a "NaN"-value. NaN stands for Not a Number and
represents specifically missing numeric values. This object is provided by the 
numpy package and works nicely with pandas:

```python
import numpy as np # do this if you have not already imported numpy
df_dwd.loc[indices_of_missing_values, "tair_2m_mean"] = np.NaN
```

Now the irritating -888.88 values are replaced and you can easily plot and resample
the data in a meaningful manner:

```python
fig = px.scatter(df_dwd, y="tair_2m_mean")
fig.show()

df_dwd_daily = df_dwd.resample(rule="d", on="date_time").mean()
fig_daily = px.scatter(df_dwd_daily, y="tair_2m_mean")
fig_daily.show()

```

### 2. Gap Filling, interpolation and modelling

In the next part we will discuss how we can work with timeseries that have gaps of different sizes.
This is a regular task when working with long-time observations and there are a couple of options,
depending on what data is available to you and what is the final evaluation goal you have in mind.

#### 2.1: Simple linear interpolation

You do basic interpolation in your every day live. You want to bake a cake and only find a receipe for an 8 person cake,
but only 3 friends are coming over for cake time. In the receipe you have to use 1 kg of flour. Intuitively you can 
see that since you will only be four at the table, you can alter the receipe and only use 500 g of flour.
And already did you do some interpolation! 
What you easily did right away in your head could be mathematically formulated as:
y = 125 * x
where y is the amount of flour in grams and x is the number of people eating cake.

The formula for an interpolation between two points (x1,y1) and (x2,y2) at a specific point
(xn, yn) is:  

<div> $$ yn = y1 + \frac{(y_{2}-y_{1})}{(x_{2}-x_{1})} * (x_{n} - x_{1}) $$ </div>

We simply construct a straight line where y1 is our y-intercept, the slope is derived 
from the two points with the well known slope-formula 

<div> $$ m = (y2-y1)/(x2-x1) $$ </div> 

and our x value on this constructed line is difference between the point we want to look at minus the starting point

Note that in this form of y = mx + b we only have one x which we use to explain our y-value. We have one "predictor".
Using only one predictor gives us a so called simple linear regression. This is a super simple form of interpolation 
and of course leaves a lot of information aside. 

Lets look at a simple example of how to actually do linear interpolation in Python:  
  
First we create a data set to play with. We create a simple running index from 1 to 11 and some 
made up y-values. We make one array in which all values are present and a second in which some
values are missing. :

```python
index = [1,2,3,4,5,6,7,8,9,10,11]
data = {
    "full_data" : [1,2,0,13,4,10,19,15,13,21,27],
    "missing_data" : [1,2,0,np.NaN,4,np.NaN,19,15,13,np.NaN,27]
}
data = pd.DataFrame(index = index, data = data)
```

lets take a quick look at the two datasets:

First we create as simple plot to look at the characteristics of the data. Lets make a 
quick little function to keep a bit of styling:  

```python
import plotly.express as px
def scatter_plot_interp(data, columns:list[str], show=True):
    fig = px.scatter(data, y=columns)
    fig.update_traces(marker_size=10)
    fig.update_layout(template="simple_white")
    if show:
      fig.show()
    return fig
scatter_plot_interp(data, ["full_data", "missing_data"])
```

Since the two plots overlay each other, you can see the "missing" values in blue and all the ones
in the reduced dataset in red.  

To do a linear interpolation between each adjacent points you can use a numpy function, np.interp().
You can find its documentation [here](https://numpy.org/doc/stable/reference/generated/numpy.interp.html).  
The function takes 3 main arguments: 
1. The x-coordinates for which the data shall be interpolated
2. The x-coordinates of the input data
3. The y-values of the input data

The catch however is that the function will return NaN if there are NaN-values in the input arrays.  
That means we have to handle the NaNs before. Specifically we will do the following:  
- find indices of NaN values
- find indices of non-NaN-values
- give ONLY the non-NaN-values as the 2nd and 3rd argument
- give only the NaN-indices as the first argument

function: 
```python
# 1. get indices of missing and present points:

indices_of_missing_points = data.loc[data["missing_data"].isna()].index
indices_of_present_points = data.loc[data["missing_data"].notna()].index

# 2. interpolate missing values and store them in 
# a new column in the dataframe:
data.loc[indices_of_missing_points,"interpolated_data"]= np.interp(indices_of_missing_points, 
          data.loc[indices_of_present_points,"missing_data"].index, 
          data.loc[indices_of_present_points,"missing_data"])
```

In this approach all we did was to draw straight lines between adjacent points. As you see, for the first point
the prediction was rather poor, the other two where pretty well reconstructed:

```python
fig = scatter_plot_interp(data, ["full_data", "interpolated_data"])
```

However, with this approach we leave all the information the other points give us about the data aside. 
Imagine for example that you have a timeseries where you measure temperature at midnight and at 12AM.
If one datapoint was missing, you would connect the two night time temperatures and interpolate the daytime
temperature way off.

A simple measure of how well our model performed is to look at the residual standard error. We calculate it
as

<div> $$ \sqrt{\frac{\sum_{i=1}^n (y[i] - y_predicted[i])^2}{df}} $$ </div>

where y is the true value, y_predicted is the predicted y value, and df is the degrees of freedom. Df is the total
number of observations used for the model fitting minus the number of model parameters. Since we have 11 total 
data points of which 3 are missing and we have 2 model parameter we have 6 degrees of freedom.  

{% capture exercise %}

<h3> Exercise </h3>
<p >Use your knowledge of pandas and numpy to write a function that returns the RSE</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
def get_RSE(y_true, y_predicted, degrees_freedom):
    RSE = np.sqrt(np.sum((y_true - y_predicted)**2) / degrees_freedom)
    return RSE

y_true = data.loc[indices_of_missing_points, "full_data"]
y_predicted = data.loc[indices_of_missing_points, "interpolated_data"]
rse = get_RSE(y_true, y_predicted, degrees_freedom=6)
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>

#### 2.2: Simple linear models

Another approach is be to create a linear model that builds not only on the two points adjacent to the one we want to know,
but rather the whole of the dataset that we have available.

So what we want to achieve, is to find a function that constructs our unknown data points based on the data we have
available in the best possible way. That means, that we want to have as little errors in our model as possible. 
The error is usually measured as the "sum of squared errors" (SSE) which is the total of distances between true values 
and the predicted values. We square it to avoid negative and positive values counterbalancing each other. 

Looking at an array of n data points we can write  

<div> $$ SSE = \sum_{i=1}^n (y(i) - b - m * x(i))^2 $$ </div> 
  
y(i) is the true y value at the predicted point, b is the y-intercept of the linear model, 
m is the first coefficient of the linear model and x(i) is the x-value at the predicted point. 

Since we want to find the straight line, that MINIMIZES the SSE, we call a procedure like this
a "minimization problem" and specifically the estimation of this line is called a "least squares estimation".

In the easiest way of fitting a linear model to such a dataset, it all depends on the mean of our dataset.
To derive the model parameters we can use the following relations where we replace b with alpha and m with beta
(as that is the general standard). Also we will now denote the predicted y-value with a ^ on top of that, which is
the common standard in literature. Sometimes this is also referred to as y_hat.  

<div> $$ \hat{y}_{i} = \alpha + \beta * x_{i} $$ </div>

<div> $$ \alpha = \bar{y} - (m \bar{x}) $$ </div>

<div> $$ \beta = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2} $$ </div>

If we would do it by hand, we would simply plug in all the numbers we have into the expression for beta and
use the result to derive our alpha

But we are working with Python so we will now introduce an awesome modelling and machine-learning library called
"scikit-learn".  
Scikit-learn has a huge amount of model-packages available, from simple linear regression all the way advanced statistical
regressions, classifications and analysis tools. [The documentation is also quite nice and extensive!](https://scikit-learn.org/stable/index.html)  
We will make use of scikit-learn to fit a simple linear regression model to our data. However to do so we need to some tweaking of our data.
Especially two things are important:  
1. Scikit learn can not work with NaN-data. That means we need to filter these out of the data we feed to the linear model
2. The model needs two-dimensional data. A list like [1,2,3] is one-dimensional and does not work with creating linear models.
Instead we have to bring it to a form of [[1],[2],[3]] etc. We can do this using the "reshape" function.

```python
from sklearn.linear_model import LinearRegression
# Scikit learn is quite object oriented. That means,
# we imported a Class called "LinearRegression", which contains
# all the following functions to work with the model.

# Step 1: instantiate the class
linearModel = LinearRegression()

# Step 2: Fit the model. This is the process of 
# feeding our known data to the model and tweaking the
# parameters so that the prediction error gets minimized.
# However, sklearn can not work with NaN values, so again
# we need to leave them out in the fitting:

# --- repetition from before:
indices_of_present_points = data.loc[data["missing_data"].notna()].index
x = data.loc[indices_of_present_points,"missing_data"].index.values.reshape(-1,1)
y = data.loc[indices_of_present_points,"missing_data"].values.reshape(-1,1)
# --- fitting the model:
linearModel.fit(x,y)

# Step 3: Check how well our model performed! sklearn has an inbuilt
# function for it called "score". It returns the R^2 value for 
# the true values and the values predicted by the model:
linearModel.score(x,y)
```  
We can obtain the paramters of the linear model, that scikit-learn has created for us:

```python
m = linearModel.coef_
b = linearModel.intercept_
print(f"Linear equation: {m}*x+{b}")
```

The last thing left to do is to use this model to predict our missing values.
All we need to do is use the models "predict()" function and give it the 
indices we want to prediction for:

```python
# first we create a new column consisting of NaN values
data["sklearn_prediction"] = np.NaN
# then we replace the values in the missing rows with our model prediction:
linear_prediction = linearModel.predict(indices_of_missing_points.values.reshape(-1,1))
data.loc[indices_of_missing_points, "sklearn_prediction"] = linear_prediction
```

We can look at out interpolated values by plotting them as red dots together with our reduced dataset:
```python
fig_linmod = scatter_plot_interp(data, ["full_data", "sklearn_prediction"], show=False)
```
As you can see, the linear model already performs a bit better than the simple linear interpolation does. 
We can visualize the linear regression line by predicting the full array of x-values and plotting
the result as a line:

```python
data["yhat_full"] = linearModel.predict(data.index.values.reshape(-1,1))
fig_linmod.add_traces(
    go.Scatter(
        x = data.index,
        y = data["yhat_full"],
        mode="lines",
        name="linear regression line"
    )
)
fig_linmod.show()
```
Finally we need to look at statistical metrics to find out, how well our linear model performed.
Luckily we can easily get a whole range of such metrics from the sklearn.metrics package. 
Lets define a simple function to grab a bunch of metrics at once:

```python

import sklearn.metrics as metrics
def regression_results(y_true, y_pred):

    # Regression metrics
    mse=metrics.mean_squared_error(y_true, y_pred) 
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('r^2: ', round(r2,4))
    print('MAE: ', round(median_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
```

A few things we can take from this are:  
**a)** r^2 is 0.8351, which is the ratio of the sum of squared errors divided by the sum of squared
deviations from the mean. You can say that r^2 is a measure of how much of the variance in the original
data is reflected by the model. In this case, as our model is just a line, the amount of variance captured in the model
stems from the linear trend that is inherent in the original data.

Whether an r^2 is reflective of a good correlation depends heavily on the application. If you are a social scientist and work on 
voter behaviour an r_square of 0.65 may be spectacularly good. If you want to calibrate your measurement device
and the reference and measured values have an r^2 of less than 0.85 you might want to check it again...

**b)** The mean squared error (MSE) is exactly that: we calculate the distance from each datapoint to its predicted counterpart,
and to avoid negative errors counterbalancing positive ones, we square them. Then we take the mean of all errors. Due to the squaring
the errors get quite high and are not directly interpretable. That is why we take the square root of the squared errors and get to the
"root mean square error" (RMSE). This is an error very often reported in model performance evaluation, also often used in scientific papers.

**c)** Lastly the median absolute error (MAE) is a different performance metric that gets shown not as often, but is still very useful.
For the MAE, we also calculate each error, take the absolute of it (make negative values positive) and then grab the median value,
so the one that sits right in the middle of all datapoints. Because we take the median instead of the mean, this metric is insensitive
to outliers. If we have very low errors, but then a few extremely high ones (or the other way around), the mean value can be skewed
while the median would not change.  

We wont go much deeper into statistical metrics here. But as you can see, this model does represent certain characteristics
of the data regarding its variance (judging by the rsquare of > 0.8)
but has a pretty high average error of more than 4 while we are in a 
domain of data that only reaches from 1 to 27.

### Part 2.3: Multiple linear models

Lets look at another way to make our models a bit more flexible
So far we created a linear model with only one parameter. Obviously that did not catch all of the variance in our 
data. In reality we often have more data at hand which can help us explain the measure of interest.
For example to fill gaps in temperature data instead of only using the indices to predict, we could add variables
such as the incoming radiation or the relative humidity of the air.

Lets continue working with the dwd data we used before. Load it just like we did in the previous exercises

When we try to simply interpolate with the pointwise linear interpolation, 
you will see that we get a pretty uninformed output.

We will now create a more sophisticated model to reconstruct our missing data. However, this time we have a whole dataset
of predictors to choose from.  
Since we want to fill a gap in temperature data, we need to find predictors that are well correlated with 
temperature. To figure out which ones are suitable we can make use of the correlation matrix.  
A correlation matrix is a normalized form of a covariance matrix. The values vary between -1 and 1. 
A value of 1 signals a perfect positive, -1 a perfect negative correlation. 0 means that the two variables are not
correlated at all. With pandas you can get the full correlation matrix with all variables with the .corr() function:

```python
# lets look at the correlation matrix
df_dwd.corr()
# you can plot and explore it with plotly.
# The interactivity is really handy here:
px.imshow(df_dwd.corr()).show()
# To get all correlations with tair_2m_mean we have to index it:
df_dwd.corr()["tair_2m_mean"]
```
Try to figure out, which variables could be suitable to fill the gaps in our data from the below table.

Now we can go ahead and start building our multivariate model! Let go!
Before we really start plugging the data into the model we need to do a bit of 
preparation:  
Since the model has to be fit with data where all the predictors we want are present AND
we have observation data of our target variable to train the model on, we first need to find that
data. We can do that easily by dropping the rows, where these columns are na with "dropna()":

```python
present_data = df_dwd.loc[:,["SWIN","rH","tair_2m_mean"]].dropna()
```

Now we want to split these into the data we use as predictors (y) and the data we want to 
predict (x, also called the "predictand"):

```python

y = present_data.loc[:,["SWIN","rH"]]
x = present_data.loc[:,["tair_2m_mean"]]
```

Finally one last very important step is that we need to split our available data into two parts:
a training and a testing dataset. The training data will ONLY be used for creating (or "fitting")
the model. To test the performance of the model, we keep a fraction of the available data out 
of the training set. That way we can predict the testing data and compare it to the real results.
We are working with some artificiallly created gaps in the data here, but in real life you would 
otherwise have no way to test, how well your model actually predicts data.  
It is extremely important to do this split, because you can never test a model on data that it 
has already seen during its training phase. That would skew your results and make it look better than
it actually is.  
Luckily, because this is such a common task to do, scikit learn has us covered with a very simple function
to do the splitting:
```python
from sklearn.model_selection import train_test_split 
# creating train and test sets 
X_train, X_test, y_train, y_test = train_test_split( 
	x, y, test_size=0.3, random_state=42) 
```

Now that we have our final training and testing datasets ready for use, we can go ahead and fit our model!

```python
linearModel = LinearRegression()
# Only training data used for fitting:
linearModel.fit(X_train,y_train)
# Only testing data used for the score:
linearModel.score(X_test,y_test)

# You can plot the prediction for the testing period
# as a scatter plot to get an idea of the spread
# of the errors. Put true values on one axis and predicted on the other:
yhat = linearModel.predict(X_test).reshape(1,-1)[0]
px.scatter(x=yhat,y=y_test["tair_2m_mean"]).show()
```
As you can see the score is roughly 0.36. That is not exactly great but does indicate
a weak correlation between predicted and true values. 

{% capture exercise %}

<h3> Exercise </h3>
<p >Do a linear interpolation and 1-D linear model prediction for this same data.
Do any of them perform equally good or better than the multiple regression? </p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
#------- interpolation:
interpolated_data = np.interp(
    y_test.index,
    y_train.index,
    y_train["tair_2m_mean"])

regression_results(y_test, interpolated_data)


#-------- 1-D linear model:
y = present_data.loc[:,["tair_2m_mean"]].values.reshape(-1,1)
x = present_data.index.values.reshape(-1,1)

from sklearn.model_selection import train_test_split 
# creating train and test sets 
X_train, X_test, y_train, y_test = train_test_split( 
	x, y, test_size=0.3, random_state=101) 

linearModel = LinearRegression()
linearModel.fit(X_train,y_train)
y_hat = linearModel.predict(X_test)
regression_results(y_test, y_hat)
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>


### 2.4: Machine Learning approaches (example Random Forests)
Here we will just take a quick look at a machine learning method which is commonly used for gap filling applications,
Random Forests.
The purpose is that you get an idea of how to implement such a method and at least you have seen it. We will not go
into the details of the actual method.



First we need an extra library for this regression:
```R
install.packages("randomForest") # <-- You can comment this line out once you installed the package
library(randomForest)
```

When applying machine learning or generally models to actual applications, what you have to do is split your data
into two independent sets. One you will use to construct your model on, this is the so called "training" dataset.
The second dataset will be used to test your constructed model on and see how well it performs on data it has 
never seen before. This split is extremely important to maintain, because otherwise you might get an overestimation
of your model performance and your claims can easily be disproved.  
In the previous examples we omitted this procedure because we knew the missing data and could evaluate our model
performance on it, in real life however we do not have the missing data and need to act like a part of the present
datapoints are missing.



From the reduced dataset we remove the rows containing NAN, which is the data we actually do not have.
```R
data_site_daily_reduced_noNA = data_site_daily_reduced[complete.cases(data_site_daily_reduced),] # Remove rows containing NA
```
Additionally the columns containing datetime, line, DP and ST are removed because they are no good predictors or 
are derived from temperature. In the case of surface temperature, we will just act like we don't have it to make
the prediction more interesting.
```R
data_site_daily_reduced_noNA = data_site_daily_reduced_noNA[3:18] # Removing line and ST
data_site_daily_reduced_noNA = data_site_daily_reduced_noNA[,-14] # Removing ST
data_site_daily_reduced_noNA = data_site_daily_reduced_noNA[,-5] # Removing DP
```

Now we will split the remaining dataset into two sets comprising of 70% of the date for training and 30% for testing
We will not use a consecutive sample of the data (e.g. the first x%) but rather a random sample. This is because
consecutive data often time contains a correlation in itself which can lead to biased models. No further details here,
just keep in mind that when training models randomizing the input is an important point (keyword "autocorrelation")

For  this we can use the sample() function. We pass it all available indices of our dataset with "nrow(dataset)". As
input you can use either a vector of values or an integer. If it is an integer like we used here, it is the values 
1 to the integer, so 1 to nrow(data_site_daily_reduced_noNA).
Then we specifiy that we want our sample to be 70% of that data with 0.7*nrow(data_site_daily_reduced_noNA).
With replace = False we specify that we want to remove the indices after sampling them, so we can not pick
an index twice in our sample.
```R
train <- sample(nrow(data_site_daily_reduced_noNA), 0.7*nrow(data_site_daily_reduced_noNA), replace = FALSE)
```


Now we use the sample of indices above to create our training and testing datasets. 
First we use the indices directly to pick the values from the original dataset into our TrainSet
```R
TrainSet <- data_site_daily_reduced_noNA[train,]
```

Then we use the same indices to create our validation dataset, by picking those indices from the original dataset
which are NOT in the train indices array by putting a minus in front of it. That is "exclusive" indexing.
```R
ValidSet <- data_site_daily_reduced_noNA[-train,]
nrow(TrainSet)
nrow(ValidSet)
```

I will only give a very brief intro to random forests here, no need to memorize that. If you are interested,
you can also look the below youtube video for a very good short video on the method.  
https://www.youtube.com/watch?v=v6VJ2RO66Ag
  
Very generally speaking you can say that this algorithm looks at your data and the predictors and it picks
a few of  the predictors, leaving others out.
With this reduced set it trains a model. That means, it tries to find out under which circumstances in the 
predictors, the data has a certain value. 
In random forests, many of those models are trained and compared. Each with different predictors and trained on 
different amounts and points of training data.
After building the model, you can use it to predict unknown values. Therefore, the predictor data for these 
unknown datapoints is fed into each of these models and the combined output from all of them is evaluated as the
final decision.

We will create a Random Forest model with mtry = 6. The mtry keyword defines, how many predictors will be considered
in each model. The argument ntree = 500 means that we will create a total number of 500 models, each containing
a different combination of predictors and data.
```R
rfmodel <- randomForest(T ~ ., data = TrainSet, importance = TRUE, replace=FALSE, mtry=6, ntree=500, type="regression")
rfmodel
```



Similar to how we used the predict() method before for the linear models, we can use it here on our random forest
model. We feed it the model and our validation dataset to test the model performance on unknown data:
```R
predValid <- predict(rfmodel, ValidSet)
```



Lets take a look at how the model output looks compared to the actual data:
```R
plot(ValidSet$$T, xlab="Day", ylab="T [°C]")
points(predValid, pch=19, col="red")
legend(110,26, legend=c("true data", "modelled data"), col=c("black", "red"), pch=c(1,19))
```



Now we calculate some metrics to evaluate our model performance: 
```R
metrics = data.frame(
    "RMSE" = sqrt(mean((ValidSet$$T - predValid)^2)),
    "R^2" = cor(ValidSet$$T, predValid)^2
    )
metrics
```

Our R^2 of more than roughly 0.78 is quite satisfying, considering that we are dealing with daily averaged data in a meteorological context.
A reasonable amount (78%) of the data variance is represented by our model.
The root mean square error of 2.9 is not too bad, but does indicate that specific absolute values are not
exactly predicted by the model.

Finally we can use our model to predict the missing data in our original dataframe:
```R
predgap <- predict(rfmodel, data_site_daily_reduced[removed_indices,])
```

To compare our results, we can plot the reduced data, the true valus and our model output together:
```R
plot(data_site_daily_reduced$$T, xlab="Day", cex=0.85, ylab="T [°C]")
points(removed_indices,predgap, pch=19, cex=0.85, col="red")

#points(removed_indices,data_site_daily$$T[removed_indices], cex=0.85, pch=19, col="black")
#legend(1,25, legend=c("gap data", "modelled data", "true data"), col=c("black", "red", "black"), pch=c(1,19, 19))
legend(1,25, legend=c("gap data", "modelled data"), col=c("black", "red"), pch=c(1,19))
```

Also look at the plot above with the true data plotted (remove the commenting sign in the cell above).
Now we calculate some metrics to evaluate our model performance: 
```R
metrics = data.frame(
    "RMSE" = sqrt(mean((data_site_daily$$T[removed_indices] - predgap)^2, na.rm=TRUE)),
    "R^2" = cor(data_site_daily$$T[removed_indices], predgap,  use="complete.obs")^2
    )
metrics
```

As you can see, the R^2 of our gap filled data is not very high. We used the model to predict a gap of data that
is a lot smaller than the validation dataset. It seems that, even though the model did a pretty good job 
predicting the trend of the data on longer timescales, when we look at shorter periods the represented variance decreases
drastically.

Keep in mind that a model is by definition NEVER the actual truth. Its goal is to come as close as possible to the truth
while often times drastically reducing the complexity of the issue. In a real world scenario we would not
know that our final modelled data is not the same as the true data. Therefore all we can do is create and test our models
to the best of our knowledge and be honest about what they are capable of doing and what their shortcomings are!
