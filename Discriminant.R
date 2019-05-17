
## This program predicts the number of floors in single family
## homes using discriminant analysis (both LDA and QDA).

setwd("/Users/Bradley/Dropbox/...")

library(MASS)

# read in data + clean
h_data <- read.csv("housing_data.csv",header = T)
str(h_data)
summary(h_data)
length(which(is.na(h_data)))
h_data <- na.omit(h_data)
rows <- nrow(h_data)
h_data$Floors <- as.factor(h_data$Floors)
h_data$Rooms <- h_data$Bed + h_data$Bath

# plot data
pairs(h_data[c("Floors","SqFt","LotSqFt","Rooms")])


############
## LDA
############
lda_fit <- lda(h_data$Floors ~ h_data$SqFt + h_data$LotSqFt + h_data$Rooms,CV=TRUE)
lda_outcome <- table(h_data$Floors, lda_fit$class)

# confusion matrix
lda_outcome

# percent correct by class
diag(prop.table(lda_outcome, 1))

# total percent correct
sum(diag(prop.table(lda_outcome)))


############
## QDA
############
qda_fit <- qda(h_data$Floors ~ h_data$SqFt + h_data$LotSqFt + h_data$Rooms,CV=TRUE)
qda_outcome <- table(h_data$Floors, qda_fit$class)

# confusion matrix
qda_outcome

# percent correct by class
diag(prop.table(qda_outcome, 1))

# total percent correct
sum(diag(prop.table(qda_outcome)))
