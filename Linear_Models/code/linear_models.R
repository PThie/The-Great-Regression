#--------------------------------------------------
# load library

library(dplyr)
library(fixest)
library(data.table)
library(here)
library(ggplot2)

#--------------------------------------------------
# paths

main_path <- file.path(here(), "Linear_Models")
data_path <- file.path(main_path, "data")

#--------------------------------------------------
# load data

housing <- data.table::fread(
    file.path(
        data_path,
        "housing.csv"
    )
)

#--------------------------------------------------
# some cleaning

housing_prep <- housing |>
    # remove observations for values larger than the 99 percentile
    # the values are somewhat clustered at the maximum
    dplyr::filter(
        median_house_value < as.numeric(
            quantile(median_house_value, probs = 0.99, na.rm = TRUE)
        )
    ) |>
    # transform the income (since it is measured in tens of thousands of US dollars)
    dplyr::mutate(median_income = median_income * 10000)

#--------------------------------------------------
# print summary

summary(housing_prep)

#--------------------------------------------------
# linear model

# fitting the linear model
linear_mod <- fixest::feols(
    median_house_value ~ median_income,
    data = housing_prep,
    se = "hetero"
)

# printing the estimated coefficients
fixest::etable(
    linear_mod,
    se = "hetero",
    signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.1)
)

# plotting the relationship between house value and number of rooms
# fitting a linar trend

# generate the predicted values
predicted_values <- data.frame(
    pred = predict(linear_mod, housing_prep), income = housing_prep$median_income
)

# restrict to maximum in original data
predicted_values <- predicted_values |> 
    dplyr::filter(pred <= max(housing_prep$median_house_value, na.rm = TRUE))

# generate plot
ggplot()+
    geom_point(
        aes(
            x = median_income,
            y = median_house_value
        ),
        data = housing_prep
    )+
    # add regression line
    geom_line(
        data = predicted_values,
        aes(x = income, y = pred),
        color = "red",
        linewidth = 1
    )+
    scale_y_continuous(labels = scales::comma)+
    scale_x_continuous(labels = scales::comma)+
    labs(
        x = "Median income (in US dollars)",
        y = "Median house value (in US dollars)"
    )+
    theme_light()

#--------------------------------------------------
# multilinear model

multilinear_mod <- fixest::feols(
    median_house_value ~ total_rooms + median_income + population + total_bedrooms + housing_median_age,
    data = housing_prep,
    se = "hetero"
)

fixest::etable(
    multilinear_mod,
    se = "hetero",
    signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.1)
)
