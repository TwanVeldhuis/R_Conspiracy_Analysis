# step 0 - Get all Packages ####
#install.packages("rstudioapi")
library(PerformanceAnalytics)
library(rstudioapi) 
library(car)

# Get the current script file location and expand the whole file location and set it
file.path <- getSourceEditorContext()$path 
dir.path <- dirname(normalizePath(file.path))
setwd(dir.path)

# step 1 - Prepare the Data ####
rm(list=ls())

# Load data (assuming it's stored in a data frame called "conspiracy")
conspiracy <- read.table('./data.txt', header=TRUE, sep="\t")

# Remove not usable columns
data <- subset(conspiracy, select=-c(weight,cons_biowpn,cons_covax,cons_biowpn_dummy,cons_covax_dummy))

# Recode trust_1 variable into a binary variables (one for trust/distrust and one for full_trust/no_full_trust)
data$distrust <- ifelse(data$trust_1 <= 2, 1, 0)

# Remove rows with missing values
data <- na.omit(data)

# step 2 - Backward Stepwise Regression and Logistic Regression ####
# define intercept-only model
summary(intercept_only <- glm(distrust ~ 1, data=data, family="binomial"))

# define model with all predictors
summary(all <- glm(distrust ~ .-trust_1, data=data, family="binomial"))

# perform backward stepwise regression
backward <- step(all, direction='backward', scope=formula(all), trace=0)

# view results of backward stepwise regression
backward$anova

# view final model
backward$coefficients

# step 3 - Check model ####
# get correlation matrix
corr_data <- data[,c("populism_2", "populism_3", "populism_4", "populism_5", 
                     "cov_beh_sum", "white", "idlg", "md_radio", "md_localtv")]
chart.Correlation(corr_data, histogram=TRUE, pch=19)

# removing the correlated predictors, based on which one have the biggest negative influence on the AIC.
# So we removed the populism_3 and populism_5 predictors.
corr_data2 <- data[,c("populism_2", "populism_4", 
                      "cov_beh_sum", "white", 
                      "idlg", "md_radio", 
                      "md_localtv", "distrust")]

# create the model based on the results
summary(lr2.model <- glm(distrust ~ ., 
                         data=corr_data2, family=binomial))
chart.Correlation(corr_data2, histogram=TRUE, pch=19)

# check vif values

# Create bar plot with rotated y-axis labels
par(mar=c(5, 7, 4, 2) + 0.1)
vif_values <- vif(lr2.model)
barplot(vif_values, main = "VIF Values", horiz = TRUE, col = "steelblue", xlim=c(0,2), las=1)
par(mar = c(5, 4, 4, 2) + 0.1)

# Now we got the predictors that have the highest statistical significance and have the lowest multi-colinearity
