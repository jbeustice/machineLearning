
## This program predicts the number of inches of rain in 
## region 9, given the number of inches of rain in surrounding
## regions using ensemble methods (stacking) from 2 base
## models: k-NN and random forests.

setwd("/Users/Bradley/Dropbox/website/machine_learning/")

library(caret)
library(randomForest)

# read in data + clean
r_data <- read.csv("rain_data.csv",header = T)
str(r_data)
r_data <- subset(r_data,select=-Date)
summary(r_data)
rows <- nrow(r_data)

# split data --> training and test
split <- sample.int(n=rows,size=floor(0.8*rows),replace=F)
training <- r_data[split, ]
test  <- r_data[-split, ]

### k-NN

# tune k-NN regression training model
cv <- trainControl(method="cv",number=5)
knn_fit <- train(subset(training,select=-RG09),training$RG09,method="knn",
                 tuneGrid=expand.grid(k=1:10),trControl=cv,metric="RMSE")
knn_fit
knn_fit$finalModel

### Random forests

# determine number of trees
rf <- randomForest(RG09~.,data=training,ntree=1000)
rf
plot(rf)
which.min(rf$mse)

# choose mtry: # of vars to consider each split
oob.err=double(15)
for(i in 1:15){
  fit=randomForest(RG09~.,data=training,mtry=i,ntree=500)
  oob.err[i]=fit$mse[500]
  cat(i," ")
}

# plot mtry data
plot(1:i,oob.err,pch=19,col="red",type="b",ylab="Mean Squared Error",
     xlab="mtry values 1 to 15",main="Out-of-bag Error")
which.min(oob.err)

### Ensemble

# partition data
folds <- sample(rep(1:5,length=nrow(training)))
table(folds)
train_meta <- training
train_meta$m1 <- NA
train_meta$m2 <- NA
test_meta <- test
test_meta$m1 <- NA
test_meta$m2 <- NA

# predict each model to training data
for(i in 1:5){
  # prep data
  train.x <- subset(training[folds!=i,],select=-RG09)
  train.y <- training$RG09[folds!=i]
  train.test <- subset(training[folds==i,],select=-RG09)
  
  # fit and predict using kNN
  knn.fit <- knnreg(train.x,train.y,k=7)
  train_meta$m1[folds==i] <- predict(knn.fit,newdata=train.test)
  
  # fit and predict using RF
  rf.fit <- randomForest(RG09~.,data=training[folds!=i,],mtry=5,ntree=500)
  train_meta$m2[folds==i] <- predict(rf.fit,newdata=train.test)
}

# fit and predict using kNN to test data
knn.test <- knnreg(subset(training,select=-RG09),training$RG09,k=7)
test_meta$m1 <- predict(knn.test,newdata=subset(test,select=-RG09))

# fit and predict using RF to test data
rf.test <- randomForest(RG09~.,data=training,mtry=5,ntree=500)
test_meta$m2 <- predict(rf.test,newdata=subset(test,select=-RG09))

## combine models (via regression)
fit.lm <- lm(RG09~m1+m2-1,data=train_meta)
summary(fit.lm)
coefi <- coef(fit.lm)
test_meta$final <- as.matrix(test_meta[,names(coefi)])%*%coefi
final.error <- apply(abs(test_meta$RG09-test_meta$final),2,mean)
final.error
