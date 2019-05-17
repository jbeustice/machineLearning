
## This program predicts the number of floors in single family
## homes using the k-nearest neighbor algorithm. The value k
## is found by cross-validation.

setwd("/Users/Bradley/Dropbox/...")

# load required libraries
library(caret)

# read in data + clean
h_data <- read.csv("housing_data.csv",header = T)
str(h_data)
summary(h_data)
length(which(is.na(h_data)))
h_data <- na.omit(h_data)
rows <- nrow(h_data)
h_data$Floors <- as.factor(h_data$Floors)
h_data$Rooms <- h_data$Bed + h_data$Bath

# split data --> training and test
split <- sample.int(n=rows,size=floor(0.8*rows),replace=F)
training <- h_data[split, ]
test  <- h_data[-split, ]

# fit training model
cv <- trainControl(method="cv",number=5)
knn_fit <- train(training[c("SqFt","LotSqFt","Rooms")],training$Floors,
             method="knn",tuneGrid=expand.grid(k=1:10),trControl=cv,metric="Accuracy")
knn_fit

# evaluate model on test data
test_pred <- predict(knn_fit,newdata=test)
confusionMatrix(test_pred,test$Floors)
