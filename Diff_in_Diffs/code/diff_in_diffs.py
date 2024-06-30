#--------------------------------------------------
# import packages

import os
from os.path import join
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

#--------------------------------------------------
# paths

main_path = Path(os.getcwd())
data_path = join(main_path, "data")
output_path = join(main_path, "output")

#--------------------------------------------------
# load data
# organ donor registration in the US from Huntington-Klein, Nick (2021) "The Effect"

# Additional infos on data:
#   Policy setting is used in: Kessler and Roth (2014): "DON'T TAKE ‘NO’ FOR AN
#       ANSWER: AN EXPERIMENT WITH ACTUAL ORGAN DONOR REGISTRATIONS"
#   treatment group: California (change donor policy (from opt-in to choice)
#       in Q3 2011)

organ_data = pd.read_csv(
    join(
        data_path,
        "organ_data.csv"
    )
)

#--------------------------------------------------
# define treatment and control

# if timing after Q3 2011, then treated (= 1) else control (= 0)
# defines treatment time
organ_data["treated_time"] = np.where(
    organ_data["Quarter"].isin(["Q32011", "Q42011", "Q12012"]),
    1,
    0
)

# if California then treated (= 1) else control (= 0)
# defines treatment region
organ_data["treated_california"] = np.where(
    organ_data["State"] == "California",
    1,
    0    
)

#--------------------------------------------------
# Difference-in-Differences by "hand"

# Calculate the average rate for all combinations of treatment time and region
mean_values_groups = organ_data.groupby(
    ["treated_time", "treated_california"]
)["Rate"].agg(np.mean).reset_index()

# Calculate the difference for control group across time
# NOTE: within group: its the difference of post-treatment vs pre-treatment time
def group_difference(group: int):
    diff = (mean_values_groups[
        (mean_values_groups["treated_california"] == group) &
        (mean_values_groups["treated_time"] == 1)
    ]["Rate"].values) - (mean_values_groups[
        (mean_values_groups["treated_california"] == group) &
        (mean_values_groups["treated_time"] == 0)
    ]["Rate"].values)
        
    return diff

# Calculate the difference for control group across time
diff_control_group = group_difference(group = 0)

# Calculate the difference for treatment group across time
diff_treatment_group = group_difference(group = 1)

# Calculate the difference-in-differences
# NOTE: This can be labelled as unconditional difference-in-difference since we
# are not controlling for confounding factors.

diff_overall = diff_treatment_group - diff_control_group
print(f"Unconditional difference-in-difference: {diff_overall[0]:.3f}")

#--------------------------------------------------
# preparation for plotting

# summarise the data by groups and time
organ_data_summarized = organ_data.groupby(
    ["treated_california", "Quarter"]
)["Rate"].agg(np.mean).reset_index()

# set the order on the x-axis
x_order  = ["Q42010", "Q12011", "Q22011", "Q32011", "Q42011", "Q12012"]

organ_data_summarized_sorted = organ_data_summarized.copy()

organ_data_summarized_sorted["Quarter"] = pd.Categorical(
    organ_data_summarized_sorted["Quarter"],
    categories = x_order,
    ordered = True
)

organ_data_summarized_sorted.sort_values("Quarter", inplace = True)

# extract the mean values for horizontal lines
y01 = mean_values_groups[
    (mean_values_groups["treated_time"] == 0) &
    (mean_values_groups["treated_california"] == 1)
]["Rate"].iloc[0]

y11 = mean_values_groups[
    (mean_values_groups["treated_time"] == 1) &
    (mean_values_groups["treated_california"] == 1)
]["Rate"].iloc[0]

y10 = mean_values_groups[
    (mean_values_groups["treated_time"] == 1) &
    (mean_values_groups["treated_california"] == 0)
]["Rate"].iloc[0]

y00 = mean_values_groups[
    (mean_values_groups["treated_time"] == 0) &
    (mean_values_groups["treated_california"] == 0)
]["Rate"].iloc[0]

#--------------------------------------------------
# visual inspection

# generate lineplot
fig, ax = plt.subplots()

ax = sns.lineplot(
    x = "Quarter",
    y = "Rate",
    hue = "treated_california",
    legend = "full",
    hue_order = [0, 1],
    palette = {0: "blue", 1: "green"},
    errorbar = None,
    data = organ_data_summarized_sorted
)

ax.axvline(x = "Q32011", color = "black")

ax.axhline(y = y01, linestyle = "--", color = "grey", xmin = 0, xmax = 0.59)
ax.axhline(y = y11, linestyle = "--", color = "grey", xmin = 0.59, xmax = 1)

ax.axhline(y = y00, linestyle = "--", color = "grey", xmin = 0, xmax = 0.59)
ax.axhline(y = y10, linestyle = "--", color = "grey", xmin = 0.59, xmax = 1)

ax.set_ylabel("Average donor rate")
ax.set_xlabel("")

plt.legend(
    loc = "upper center",
    bbox_to_anchor = (0.5, -0.1),
    frameon = False,
    ncol = 2,
    labels = ["Control (= 0)", "Treated (= 1)"]
)

plt.show()

fig.savefig(
    join(
        output_path,
        "diff_in_diff_figure.png"
    ),
    bbox_inches = "tight",
    dpi = 400
)

#--------------------------------------------------
# Difference-in-Differences by modelling

# add interaction term to date
organ_data["interaction_time_california"] = organ_data["treated_time"] * organ_data["treated_california"]

# define variables
y = organ_data[["Rate"]]
x = organ_data[["treated_california", "treated_time", "interaction_time_california"]]

# adding a constant to the linear model
x = sm.add_constant(x)

# fit the model
did_model = sm.OLS(y, x).fit(cov_type = "HC3")

# print results
did_model.summary()

# coefficient of interaction = the overall difference in means

# Interpretation:
    # Switching from a opt-in system to a system of choice did decrease the donor
    # rate in California by 2.2 percentage points.
    # However, the effect is not significant. Therefore, it is actually a null effect
    # (the policy did not change the donor rate).