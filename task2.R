# step 0 - Get all Packages ####
#install.packages("rstudioapi")
library(rstudioapi)
library(PerformanceAnalytics)
library(car)
library(randomForest)
library(e1071)
library(caret)
library(kernlab)

# Get the current script file location and expand the whole file location and set it
file.path <- getSourceEditorContext()$path 
dir.path <- dirname(normalizePath(file.path))
setwd(dir.path)

#make this example reproducible
set.seed(1)

# step 1 - Import the data ####
rm(list=ls())

# Load data (assuming it's stored in a data frame called "conspiracy")
data.raw <- read.table('./data.txt', header=TRUE, sep="\t")

# Recode trust_1 variable into a binary variable (one for full_trust/no_full_trust)
data.raw$full_trust <- as.factor(ifelse(data.raw$trust_1 == 4, 1, 0))

# Remove not usable columns
data <- subset(data.raw, select=-c(weight,cons_biowpn,cons_covax,cons_biowpn_dummy,cons_covax_dummy, trust_1))

# step 2 - Inspect the data ####
dim(data)
str(data)
summary(data)
colnames(data)
head(data, n = 10)
tail(data, n = 5)
table(data['full_trust'])
sapply(subset(data, select = -c(full_trust)), var)
sapply(subset(data, select = -c(full_trust)), sd)
#View(data)

# based on this inspection we only have to check if maybe cov_beh_sum and age need some kind of normalization
hist(data$cov_beh_sum, main = "cov_beh_sum Distribution", xlab = "cov_beh_sum", ylab = "Frequency")
hist(data$age, main = "Age Distribution", xlab = "Age", ylab = "Frequency")

# there are no extreme outliers, so I don't think this is needed and we have to treat any of these

# step 3 - Prepare the data ####
# Remove rows with missing values
data.staging <- na.omit(data)

# next to this we also have to look for correlation in the dataframe itself
#chart.Correlation(data.staging, histogram=TRUE, pch=19)
colnames(data.staging)

# then the following columns are deleted because of correlation that is higher than 0.3
data.staging2 <- subset(data.staging, select=-c(populism_1, populism_5, pid2, pid3, md_radio, md_national, md_broadcast, md_con, md_agg, ms_news, rw_news))
#chart.Correlation(data.staging2, histogram=TRUE, pch=19) # now there are no more correlations higher than 0.3

# finally the data is clean
data.clean <- data.staging2

# prepare the data for test and training
# so for all the models the testing and training data is the same
selection <- sample(1:nrow(data.clean), size = 0.7 * nrow(data.clean))
data.train <- data.clean[selection, ]
data.test <- data.clean[-selection, ]

# remove all the staging and only keep the clean and raw data
rm(list=setdiff(ls(), c("data.raw", "data.clean", "data.train", "data.test")))

# Logistic Regression ####
# define intercept-only model
summary(intercept_only <- glm(full_trust ~ 1, data=data.clean, family="binomial"))

# define model with all predictors
summary(all <- glm(full_trust ~ ., data=data.clean, family="binomial"))

# perform backward stepwise regression
backward <- step(all, direction='backward', scope=formula(all), trace=0)

# view results of backward stepwise regression
backward$anova

# view final model
backward$coefficients

# Extract the list of selected variables from the terms attribute
selected_vars <- attr(backward$terms, "term.labels")
dependent_vars <- "full_trust"

# Fit a glm model using the selected variables
summary(lr.model <- glm(formula(paste(dependent_vars, "~", paste(selected_vars, collapse = "+"))),
                        data = data.clean,
                        family = "binomial"))

# get mcfadden R^2
library(pscl)
pR2(lr.model)

# get correlation matrix
library(PerformanceAnalytics)
corr_data <- data.clean[, selected_vars]
chart.Correlation(corr_data, histogram=TRUE, pch=19)

# check vif values
# Create bar plot with rotated y-axis labels
par(mar=c(5, 7, 4, 2) + 0.1)
barplot(vif(lr.model), main = "VIF Values", horiz = TRUE, col = "steelblue", xlim=c(0,2), las=1)
par(mar = c(5, 4, 4, 2) + 0.1)

# Fit the logistic regression model
lr.model <- glm(formula(paste(dependent_vars, "~", paste(selected_vars, collapse = "+"))), 
             data = data.train, family = "binomial")

# Make predictions on the test data
lr.pred <- predict(lr.model, newdata = subset(data.test, select = -c(full_trust)))
lr.pred <- as.factor(ifelse(lr.pred > 0.5, 1, 0))

# Calculate accuracy, precision, recall, and F1-score manually
lr.conf_mat <- confusionMatrix(data = lr.pred, reference = data.test$full_trust)$table
lr.accuracy <- sum(diag(unlist(lr.conf_mat)) / sum(lr.conf_mat))
lr.precision <- lr.conf_mat[2,2] / sum(lr.conf_mat[,2])
lr.recall <- lr.conf_mat[2,2] / sum(lr.conf_mat[2,])
lr.f1_score <- 2 * lr.precision * lr.recall / (lr.precision + lr.recall)

# Print evaluation metrics
cat("Accuracy:", lr.accuracy, "\n")
cat("Precision:", lr.precision, "\n")
cat("Recall:", lr.recall, "\n")
cat("F1-score:", lr.f1_score, "\n")

# Random Forest ####
# Train a random forest model for selection of the most important variables
rf_model <- randomForest(as.factor(full_trust) ~ ., data = data.clean, importance = TRUE)
varImpPlot(rf_model, main = "Random Forest Feature Importance Plot", type = 1)

# To be able to predict the full_trust better, in this case we will use 11 variables
# This, because there are 3 specific variables that are clearly NOT significant for the prediction
# Fit a random forest model
rf.model <- randomForest(as.factor(full_trust) ~ ., data = subset(data.train, select = -c(hispanic, white, gender)), ntree = 500)

# Predict on test data
rf.pred <- predict(rf.model, newdata = subset(data.test, select = -c(full_trust)))

# Calculate accuracy, precision, recall, and F1-score manually
rf.conf_mat <- confusionMatrix(data = rf.pred, reference = data.test$full_trust)$table
rf.accuracy <- sum(diag(unlist(rf.conf_mat)) / sum(rf.conf_mat))
rf.precision <- rf.conf_mat[2,2] / sum(rf.conf_mat[,2])
rf.recall <- rf.conf_mat[2,2] / sum(rf.conf_mat[2,])
rf.f1_score <- 2 * rf.precision * rf.recall / (rf.precision + rf.recall)

# Print evaluation metrics
cat("Accuracy:", rf.accuracy, "\n")
cat("Precision:", rf.precision, "\n")
cat("Recall:", rf.recall, "\n")
cat("F1-score:", rf.f1_score, "\n")

# SVM ####
# Create an SVM model using linear kernel
model <- svm(full_trust ~ ., data = data.clean, kernel = "linear")

# Use RFE to select the features
ctrl <- rfeControl(method = "cv", number = 5, verbose = FALSE)
svmProfile <- rfe(x = subset(data.train, select = -c(full_trust)), y = as.numeric(data.train$full_trust),
                  sizes = c(1:ncol(data.train)-1), rfeControl = ctrl,
                  method = "svmLinear", preProc = c("center", "scale"))

# Print the selected features and their rank
predictors(svmProfile)

# Train SVM model
svm.model <- svm(formula(paste("full_trust ~", paste(svmProfile$optVariables, collapse = "+"))), 
                 data = data.train, kernel = "linear", cost = 1)

# Predict on test data
svm.pred <- predict(svm.model, subset(data.test, select = -c(full_trust)))

# Calculate accuracy, precision, recall, and F1-score manually
svm.conf_mat <- confusionMatrix(data = svm_pred, reference = data.test$full_trust)$table
svm.accuracy <- sum(diag(unlist(svm.conf_mat)) / sum(svm.conf_mat))
svm.precision <- svm.conf_mat[2,2] / sum(svm.conf_mat[,2])
svm.recall <- svm.conf_mat[2,2] / sum(svm.conf_mat[2,])
svm.f1_score <- 2 * svm.precision * svm.recall / (svm.precision + svm.recall)

# Print evaluation metrics
cat("Accuracy:", svm.accuracy, "\n")
cat("Precision:", svm.precision, "\n")
cat("Recall:", svm.recall, "\n")
cat("F1-score:", svm.f1_score, "\n")

# Cross Validation - Logistic Regression ####
# define intercept-only model
summary(intercept_only <- glm(full_trust ~ 1, data=data.clean, family="binomial"))

# define model with all predictors
summary(all <- glm(full_trust ~ ., data=data.clean, family="binomial"))

# perform backward stepwise regression
backward <- step(all, direction='backward', scope=formula(all), trace=0)

# view results of backward stepwise regression
backward$anova

# view final model
backward$coefficients

# Extract the list of selected variables from the terms attribute
selected_vars <- attr(backward$terms, "term.labels")

# Set the number of folds
K = 5

# Create a matrix to store the results
cv.results <- matrix(NA, nrow = K, ncol = 2)

# Shuffle the row indices randomly and split into K folds
shuffled_indices <- sample(seq_len(nrow(data.clean)))
folds <- cut(shuffled_indices, breaks = K, labels = FALSE)

# Loop through the folds and fit the model
for (i in 1:K) {
  # Split the data into training and validation sets
  validation_indexes <- which(folds == i, arr.ind = TRUE)
  validation_set <- data.clean[validation_indexes, ]
  training_set <- data.clean[-validation_indexes, ]
  
  # Fit the model using the training set
  model <- glm(formula(paste("full_trust ~", paste(selected_vars, collapse = "+"))), 
               data = training_set, family = binomial)
  
  # Make predictions on the validation set
  predicted <- predict(model, subset(validation_set, select = -c(full_trust)), type = "response")
  predicted <- as.factor(ifelse(predicted > 0.5, 1, 0))
  
  # Calculate the accuracy and F1-score
  conf_mat <- confusionMatrix(data = predicted, reference = validation_set$full_trust)$table
  accuracy <- sum(diag(unlist(conf_mat)) / sum(conf_mat))
  precision <- conf_mat[2,2] / sum(conf_mat[,2])
  recall <- conf_mat[2,2] / sum(conf_mat[2,])
  f1_score <- 2 * precision * recall / (precision + recall)
  
  # Store the results in the matrix
  cv.results[i, 1] <- accuracy
  cv.results[i, 2] <- f1_score
}

# Calculate the average accuracy and F1-score
lr.mean_accuracy <- mean(cv.results[, 1])
lr.std_accuracy <- sd(cv.results[, 1])
lr.mean_f1_score <- mean(cv.results[, 2])
lr.std_f1_score <- sd(cv.results[, 2])

# Print the results
cat("Mean accuracy:", mean_accuracy, "\n")
cat("Standard Deviation accuracy:", std_accuracy, "\n")
cat("Mean F1-score:", mean_f1_score, "\n")
cat("Standard Deviation accuracy:", std_f1_score, "\n")

# Cross Validation - Random Forest ####
# Set the number of folds
K = 5

# Create a matrix to store the results
cv.results <- matrix(NA, nrow = K, ncol = 2)

# Shuffle the row indices randomly and split into K folds
shuffled_indices <- sample(seq_len(nrow(data.clean)))
folds <- cut(shuffled_indices, breaks = K, labels = FALSE)

# Loop through the folds and fit the model
for (i in 1:K) {
  # Split the data into training and validation sets
  validation_indexes <- which(folds == i, arr.ind = TRUE)
  validation_set <- data.clean[validation_indexes, ]
  training_set <- data.clean[-validation_indexes, ]
  
  # Fit the model using the training set
  model <- randomForest(as.factor(full_trust) ~ ., data = subset(training_set, select = -c(hispanic, white, gender)), ntree = 500)
  
  # Make predictions on the validation set
  predicted <- predict(model, newdata = subset(validation_set, select = -c(full_trust)))
  
  # Calculate the accuracy and F1-score
  conf_mat <- confusionMatrix(data = predicted, reference = validation_set$full_trust)$table
  accuracy <- sum(diag(unlist(conf_mat)) / sum(conf_mat))
  precision <- conf_mat[2,2] / sum(conf_mat[,2])
  recall <- conf_mat[2,2] / sum(conf_mat[2,])
  f1_score <- 2 * precision * recall / (precision + recall)
  
  # Store the results in the matrix
  cv.results[i, 1] <- accuracy
  cv.results[i, 2] <- f1_score
}

# Calculate the average accuracy and F1-score
rf.mean_accuracy <- mean(cv.results[, 1])
rf.std_accuracy <- sd(cv.results[, 1])
rf.mean_f1_score <- mean(cv.results[, 2])
rf.std_f1_score <- sd(cv.results[, 2])

# Print the results
cat("Mean accuracy:", mean_accuracy, "\n")
cat("Standard Deviation accuracy:", std_accuracy, "\n")
cat("Mean F1-score:", mean_f1_score, "\n")
cat("Standard Deviation accuracy:", std_f1_score, "\n")

# Cross Validation - SVM ####
model <- svm(full_trust ~ ., data = data.clean, kernel = "linear")

# Use RFE to select the top 5 features
ctrl <- rfeControl(method = "cv", number = 5, verbose = FALSE)
svmProfile <- rfe(x = subset(data.train, select = -c(full_trust)), y = as.numeric(data.train$full_trust),
                  sizes = c(1:ncol(data.train)-1), rfeControl = ctrl,
                  method = "svmLinear", preProc = c("center", "scale"))

# Print the selected features and their rank
predictors(svmProfile)

# Set the number of folds
K = 5

# Create a matrix to store the results
cv.results <- matrix(NA, nrow = K, ncol = 2)

# Shuffle the row indices randomly and split into K folds
shuffled_indices <- sample(seq_len(nrow(data.clean)))
folds <- cut(shuffled_indices, breaks = K, labels = FALSE)

# Loop through the folds and fit the model
for (i in 1:K) {
  # Split the data into training and validation sets
  validation_indexes <- which(folds == i, arr.ind = TRUE)
  validation_set <- data.clean[validation_indexes, ]
  training_set <- data.clean[-validation_indexes, ]
  
  # Train SVM model
  model <- svm(formula(paste("full_trust ~", paste(svmProfile$optVariables, collapse = "+"))), 
                   data = training_set, kernel = "linear", cost = 1)
  
  # Predict on test data
  predicted <- predict(model, subset(validation_set, select = -c(full_trust)))
  
  # Calculate the accuracy and F1-score
  conf_mat <- confusionMatrix(data = predicted, reference = validation_set$full_trust)$table
  accuracy <- sum(diag(unlist(conf_mat)) / sum(conf_mat))
  precision <- conf_mat[2,2] / sum(conf_mat[,2])
  recall <- conf_mat[2,2] / sum(conf_mat[2,])
  f1_score <- 2 * precision * recall / (precision + recall)
  
  # Store the results in the matrix
  cv.results[i, 1] <- accuracy
  cv.results[i, 2] <- f1_score
}

# Calculate the average accuracy and F1-score
svm.mean_accuracy <- mean(cv.results[, 1])
svm.std_accuracy <- sd(cv.results[, 1])
svm.mean_f1_score <- mean(cv.results[, 2])
svm.std_f1_score <- sd(cv.results[, 2])

# Print the results
cat("Mean accuracy:", mean_accuracy, "\n")
cat("Standard Deviation accuracy:", std_accuracy, "\n")
cat("Mean F1-score:", mean_f1_score, "\n")
cat("Standard Deviation accuracy:", std_f1_score, "\n")

# Plot the data ####
# Create a data frame with the mean accuracy and F1 scores for each model
results <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "Support Vector Machine"),
  Mean_Accuracy = c(lr.mean_accuracy, rf.mean_accuracy, svm.mean_accuracy),
  Std_Accuracy = c(lr.std_accuracy, rf.std_accuracy, svm.std_accuracy),
  Mean_F1_Score = c(lr.mean_f1_score, rf.mean_f1_score, svm.mean_f1_score),
  Std_F1_Score = c(lr.std_f1_score, rf.std_f1_score, svm.std_f1_score)
)

# Set font size
par(cex.lab = 1.5, cex.axis = 1.5, cex.main = 1.5)

# Create bar plot
barplot(
  c(lr.mean_accuracy, rf.mean_accuracy, svm.mean_accuracy),
  ylim = c(0, 1),
  ylab = "Accuracy",
  names.arg = c("Logistic Regression", "Random Forest", "Support Vector Machine"),
  col = c("#F8766D", "#00BA38", "#619CFF"),
  main = "Comparison of Accuracy Across Models"
)

# Add error bars
arrows(
  x0 = c(1, 2.2, 3.4),
  y0 = c(lr.mean_accuracy, rf.mean_accuracy, svm.mean_accuracy) - c(lr.std_accuracy, rf.std_accuracy, svm.std_accuracy),
  x1 = c(1, 2.2, 3.4),
  y1 = c(lr.mean_accuracy, rf.mean_accuracy, svm.mean_accuracy) + c(lr.std_accuracy, rf.std_accuracy, svm.std_accuracy),
  code = 3,
  angle = 90,
  length = 0.1
)

# Add F1 scores as text labels
text(
  x = c(0.7, 1.9, 3.1),
  y = c(lr.mean_accuracy, rf.mean_accuracy, svm.mean_accuracy),
  label = c(sprintf("Accuracy=%.2f", lr.mean_accuracy), sprintf("Accuracy=%.2f", rf.mean_accuracy), sprintf("Accuracy=%.2f", svm.mean_accuracy)),
  pos = 3
)

# Set font size
par(cex.lab = 1.5, cex.axis = 1.5, cex.main = 1.5)

# Create bar plot
barplot(
  c(lr.mean_f1_score, rf.mean_f1_score, svm.mean_f1_score),
  ylim = c(0, 1),
  ylab = "f1_score",
  names.arg = c("Logistic Regression", "Random Forest", "Support Vector Machine"),
  col = c("#F8766D", "#00BA38", "#619CFF"),
  main = "Comparison of f1 score Across Models"
)

# Add error bars
arrows(
  x0 = c(1, 2.2, 3.4),
  y0 = c(lr.mean_f1_score, rf.mean_f1_score, svm.mean_f1_score) - c(lr.std_f1_score, rf.std_f1_score, svm.std_f1_score),
  x1 = c(1, 2.2, 3.4),
  y1 = c(lr.mean_f1_score, rf.mean_f1_score, svm.mean_f1_score) + c(lr.std_f1_score, rf.std_f1_score, svm.std_f1_score),
  code = 3,
  angle = 90,
  length = 0.1
)

# Add F1 scores as text labels
text(
  x = c(0.7, 1.9, 3.1),
  y = c(lr.mean_f1_score, rf.mean_f1_score, svm.mean_f1_score),
  label = c(sprintf("F1=%.2f", lr.mean_f1_score), sprintf("F1=%.2f", rf.mean_f1_score), sprintf("F1=%.2f", svm.mean_f1_score)),
  pos = 3
)