#--------------------------------------------------
# load library

library(dplyr)
library(fixest)
library(data.table)
library(here)
library(ggplot2)

#--------------------------------------------------
# paths

main_path <- file.path(here(), "Diff_in_Diffs")
data_path <- file.path(main_path, "data")

#--------------------------------------------------
# load data
# organ donor registration in the US from Huntington-Klein, Nick (2021) "The Effect"

# Additional infos on data:
#   Policy setting is used in: Kessler and Roth (2014): "DON'T TAKE ‘NO’ FOR AN
#       ANSWER: AN EXPERIMENT WITH ACTUAL ORGAN DONOR REGISTRATIONS"
#   treatment group: California (change donor policy (from opt-in to choice)
#       in Q3 2011)

organ_data <- data.table::fread(
    file.path(
        data_path,
        "organ_data.csv"
    )
)

#--------------------------------------------------
# define treatment and control

organ_data <- organ_data |>
    dplyr::mutate(
        # if timing after Q3 2011, then treated (= 1) else control (= 0)
        treated_time = dplyr::case_when(
            Quarter %in% c("Q32011", "Q42011", "Q12012") ~ 1,
            TRUE ~ 0
        ),
        # if California then treated (= 1) else control (= 0)
        treated_california = dplyr::case_when(
            State == "California" ~ 1,
            TRUE ~ 0
        )
    )

#--------------------------------------------------
# Difference-in-Differences by "hand"

##### Calculate the average for the control group after and before the event
mean_control_after <- organ_data |>
    dplyr::filter(treated_time == 1 & treated_california == 0) |>
    dplyr::summarise(mean_rate = mean(Rate, na.rm = TRUE)) |>
    pull()

mean_control_before <- organ_data |>
    dplyr::filter(treated_time == 0 & treated_california == 0) |>
    dplyr::summarise(mean_rate = mean(Rate, na.rm = TRUE)) |>
    pull()

# take the difference between after and before average
diff_control <- mean_control_after - mean_control_before

##### Calculate the average for the treatment group after and before the event
mean_treated_after <- organ_data |>
    dplyr::filter(treated_time == 1 & treated_california == 1) |>
    dplyr::summarise(mean_rate = mean(Rate, na.rm = TRUE)) |>
    pull()

mean_treated_before <- organ_data |>
    dplyr::filter(treated_time == 0 & treated_california == 1) |>
    dplyr::summarise(mean_rate = mean(Rate, na.rm = TRUE)) |>
    pull()

# take the difference between after and before average
diff_treated <- mean_treated_after - mean_treated_before

##### Take the overall difference
# NOTE: This can be labelled as unconditional difference-in-difference since we
# are not controlling for confounding factors.
diff_overall <- diff_treated - diff_control

#--------------------------------------------------
# visual inspection

# prepare data
# summarise by groups over time
organ_data_summarized <- organ_data |>
    # set the time as factor
    dplyr::mutate(
        Quarter = factor(Quarter, levels = c(
            "Q42010", "Q12011", "Q22011", "Q32011", "Q42011", "Q12012"
        )),
        treated_california = as.factor(treated_california)
    ) |>
    dplyr::group_by(treated_california, Quarter) |>
    dplyr::summarise(
        mean_rate = mean(Rate, na.rm = TRUE)
    )

# generate plot
did_plot <- ggplot()+
    geom_line(
        data = organ_data_summarized,
        aes(
            x = Quarter,
            y = mean_rate,
            group = treated_california,
            col = treated_california
        ),
        linewidth = 1
    )+
    # define color scheme for the lines
    scale_color_manual(
        name = "",
        values = c(
            "0" = "blue",
            "1" = "darkgreen"
        ),
        labels = c(
            "0" = "Control (= 0)",
            "1" = "Treated (= 1)"
        )
    )+
    # add horizontal lines for means in control
    geom_segment(
        aes(
            y = mean_control_after,
            yend = mean_control_after,
            x = as.factor("Q32011"),
            xend = as.factor("Q12012")
        ),
        linetype = "dashed",
        linewidth = 1
    )+
    geom_segment(
        aes(
            y = mean_control_before,
            yend = mean_control_before,
            x = as.factor("Q42010"),
            xend = as.factor("Q32011")
        ),
        linetype = "dashed",
        linewidth = 1
    )+
    # add horizontal lines for means in treated
    geom_segment(
        aes(
            y = mean_treated_after,
            yend = mean_treated_after,
            x = as.factor("Q32011"),
            xend = as.factor("Q12012")
        ),
        linetype = "dashed",
        linewidth = 1
    )+
    geom_segment(
        aes(
            y = mean_treated_before,
            yend = mean_treated_before,
            x = as.factor("Q42010"),
            xend = as.factor("Q32011")
        ),
        linetype = "dashed",
        linewidth = 1
    )+
    # vertical line for event
    geom_vline(
        xintercept = "Q32011"
    )+
    # axis labelling
    labs(
        x = "",
        y = "Average donor rate"
    )+
    # overall theme of the plot
    theme_light()+
    # adjust some formatting of the plot
    theme(
        axis.title = element_text(size = 16),
        axis.text = element_text(size = 15),
        legend.key.size = unit(0.7, "cm"),
        legend.text = element_text(size = 15),
        legend.position = "bottom"
    )

# print plot
did_plot

# export plot (if needed)
# ggsave(
#     plot = did_plot,
#     "YOUR_PATH",
#     dpi = 300
# )

#--------------------------------------------------
# using regression

reg_model <- fixest::feols(
    Rate ~ treated_time + treated_california + treated_time * treated_california,
    data = organ_data,
    se = "hetero"
)

# print results
etable(
    reg_model,
    signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.1),
    se = "hetero"
)

# coefficient of interaction = the overall difference in means

# Interpretation:
    # Switching from a opt-in system to a system of choice did decrease the donor
    # rate in California by 2.2 percentage points.
    # However, the effect is not significant. Therefore, it is actually a null effect
    # (the policy did not change the donor rate).

#--------------------------------------------------
# regression with fixed effects

# It can be shown that including fixed effects results in significant effects
# (because of uncontrolled heterogeneity)
# NOTE: This does not change the coefficient size but only impacts the standard
# errors.

reg_model_fe <- fixest::feols(
    Rate ~ treated_time + treated_california + treated_time * treated_california
    # add fixed effects
    | State + Quarter,
    data = organ_data,
    se = "hetero"
)

# print results
etable(
    reg_model_fe,
    signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.1),
    se = "hetero"
)

# Interpretation:
    # Now the coefficient is significant and the law change results indeed in
    # a decrease of donor rates by 2.2 percentage points in California.