
## This program predicts the overall player score of soccor players in
## the FIFA 2019 database using random forests and boosting.

setwd("/Users/Bradley/Dropbox/website/machine_learning/")

## load required libraries
library(randomForest)
library(gbm)

# load and clean data
f_dataAll <- read.csv("fifa_data.csv",header = T)
str(f_dataAll)
f_data <- f_dataAll[,c(8,4,15,27,28,55:(ncol(f_dataAll)-1))]
f_data <- na.omit(f_data)
f_data$Weight <- as.character(f_data$Weight)
f_data$Weight <- as.numeric(substr(f_data$Weight,1,nchar(f_data$Weight)-3))
f_data$Height <- as.character(f_data$Height)
f_data$Height <- (as.numeric(substr(f_data$Height,1,1)) * 12) + as.numeric(substr(f_data$Height,3,5))
f_data$Preferred.Foot <- droplevels(f_data$Preferred.Foot)
str(f_data)
rows <- nrow(f_data)

# split data --> training and test
split <- sample.int(n=rows,size=floor(0.8*rows),replace=F)
training <- f_data[split, ]
test  <- f_data[-split, ]

# plot subset of data to see general correlations
look1 <- test[,1:10]
look2 <- test[,c(1,11:20)]
look3 <- test[,c(1,21:ncol(test))]
pairs(look1)
pairs(look2)
pairs(look3)

############
## Random forests
############

# determine number of trees
rf <- randomForest(Overall~.,data=training)
rf
plot(rf)
which.min(rf$mse)

# choose mtry: # of vars to consider each split
oob.err=double(20)
for(i in 1:20){
  fit=randomForest(Overall~.,data=training,mtry=i,ntree=400)
  oob.err[i]=fit$mse[400]
  cat(i," ")
}

# plot mtry data
plot(1:i,oob.err,pch=19,col="red",type="b",ylab="Mean Squared Error",
     xlab="mtry values 1 to 20",main="Out-of-bag Error")
which.min(oob.err)

# validate on test data with tunned parameters
rf_test <- randomForest(Overall~.,data=training,xtest=test[,-1],
                        ytest=test$Overall,mtry=7,ntree=500,importance=T)
rf_test

# view variable importance
varImpPlot(rf_test)


############
## Boosting
############

# create grid to tune parameters
hyper_grid <- expand.grid(shrinkage=c(.001,.01,.1),interaction.depth=c(1,3,5),
                          n.minobsinnode=c(10,20))

# grid search 
for(i in 1:nrow(hyper_grid)){
  
  # train model
  boost.tune <- gbm(Overall~.,data=training,distribution="gaussian",n.trees=10000,
                    interaction.depth=hyper_grid$interaction.depth[i],
                    shrinkage=hyper_grid$shrinkage[i],
                    n.minobsinnode=hyper_grid$n.minobsinnode[i],train.fraction=.75)
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(boost.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(boost.tune$valid.error))
  cat(i," ")
}

# view top models
hyper_grid[order(hyper_grid$min_RMSE),]

# build best model and validate on test data
boost <- gbm(Overall~.,data=training,distribution="gaussian",n.trees=4600,
             shrinkage=0.1,interaction.depth=5)
summary(boost)

# plot RMSE results
num.trees=seq(from=100,to=4600,by=100)
test.pred <- predict(boost,newdata=test,n.trees=num.trees)
test.err <- with(test,apply(sqrt((test.pred-Overall)^2),2,mean))
plot(num.trees,test.err,pch=19,col="red",type="b",ylab="Root Mean Squared Error",
     xlab="# Trees",main="Boosting Test Error")

# compare best model to test data
which.min(test.err)
min(test.err)

# best model RMSE for test data
test.err[length(num.trees)]
