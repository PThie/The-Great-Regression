#--------------------------------------------------
# import packages

import os
from os.path import join
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import numpy as np
from tabulate import tabulate # for nice formatting printing
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

#--------------------------------------------------
# paths

main_path = Path(os.getcwd()).parents[0] # go one directory up from current folder
data_path = join(main_path, "data")
output_path = join(main_path, "output")

#--------------------------------------------------
# load data

housing = pd.read_csv(
    join(
        data_path,
        "housing.csv"
    )
)

#--------------------------------------------------
# some cleaning

# remove observations for values larger than the 99 percentile
# the values are somewhat clustered at the maximum
housing_prep = housing[
    housing["median_house_value"] < np.quantile(housing["median_house_value"], q = 0.99)
]

# transform the income (since it is measured in tens of thousands of US dollars)
housing_prep["median_income"] = housing_prep["median_income"] * 10000

# remove rows with missing data (causes conflict in estimation)
housing_prep = housing_prep.dropna()

#--------------------------------------------------
# Look at the data

# table beginning
housing_prep.head()

# summary statistics
summary_stats = np.round(housing_prep.describe(), 2).T[['count','mean', 'std', 'min', 'max']]

# print output nicely formatted
print(tabulate(summary_stats, headers = summary_stats.columns))

#--------------------------------------------------
# RMSE function

def rmse(predicted_values, actual_values):
    return np.sqrt(((predicted_values - actual_values) ** 2).mean())

###################################################
# Linear Model
###################################################

#--------------------------------------------------
# specify test and training data

# defining our dependent and independent variable
y = housing_prep[["median_house_value"]]
x = housing_prep[["median_income"]]

# adding a constant such that it shows up in the regression
x = sm.add_constant(x)

# splitting into training and test data
x_train, x_test, y_train, y_test = train_test_split(
      x,
      y,
      test_size = 0.2,
      random_state = 0
)

#--------------------------------------------------
# run the linear regression model

# model specification
linear_model = sm.OLS(y_train, x_train)

# fitting the model
linear_model_fitted = linear_model.fit(cov_type = "HC3")

# print results
linear_model_fitted.summary()

# generate prediction
predicted_values = linear_model_fitted.predict(x_test)

# calculate RMSE
RMSE_linear_model = rmse(
    predicted_values = predicted_values,
    actual_values = y_test["median_house_value"].tolist()
)

#--------------------------------------------------
# relationship between housing values and income

# predict values from linear model
pred = pd.DataFrame({
    "pred_house_value": predicted_values,
    "median_income": x_test["median_income"]
})

# restrict to maximum price as in the original data
pred = pred[
    pred["pred_house_value"] <= np.max(housing_prep["median_house_value"])
]

# generate plot
fig, ax = plt.subplots()
sns.set_style("darkgrid")

# scatter plot
linear_trend_plot = sns.scatterplot(
    data = housing_prep,
    x = "median_income",
    y = "median_house_value",
    color = "black",
    edgecolors = "none",
    linewidth = 0,
    ax = ax
)

# line plot
trend_line = sns.lineplot(
    data = pred,
    x = "median_income",
    y = "pred_house_value",
    color = "red",
    ax = ax    
)

# define axis
ax.set(
    xlabel = "Median income (in US dollars)",
    ylabel = "Median house value (in US dollars)"    
)

# add comma separator to axis
ax.get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))

# figure size and space around it
fig.set_size_inches([8, 5])
plt.subplots_adjust(left = 0.2,bottom = 0.2, top = 0.9, right = 0.9)

# export
fig.savefig(
    join(
        output_path,
        "linear_model_plot.png"
    ),
    dpi = 400
)

###################################################
# Multilinear Model
###################################################

#--------------------------------------------------
# specify test and training data

# defining our dependent and independent variable
y = housing_prep[["median_house_value"]]
x = housing_prep[["housing_median_age", "total_rooms", "total_bedrooms", "population", "median_income"]]

# adding a constant such that it shows up in the regression
x = sm.add_constant(x)

# splitting into training and test data
x_train, x_test, y_train, y_test = train_test_split(
      x,
      y,
      test_size = 0.2,
      random_state = 0
)

# model specification
multi_linear_model = sm.OLS(y_train, x_train)

# fitting the model
multi_linear_model_fitted = multi_linear_model.fit(cov_type = "HC3")

# print results
multi_linear_model_fitted.summary()

# generate prediction
predicted_values = multi_linear_model_fitted.predict(x_test)

# calculate RMSE
RMSE_multi_linear_model = rmse(
    predicted_values = predicted_values,
    actual_values = y_test["median_house_value"].tolist()
)