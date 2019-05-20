
## This program predicts the price of single family homes
## using a variety of methods: Kmeans clustering, support vector
## machines, cross-validation, and regression (ridge, lasso, stepwise).

rm(list=ls())

setwd("/Users/Bradley/Dropbox/website/machine_learning/")

# load required libraries
library(ggplot2) # plot
library(dummies) # dummy
library(e1071) # svm
library(glmnet) # ridge, lasso
library(leaps) # stepwise

# load training data and modify/drop for analysis
h_data <- read.csv("housing_data.csv",header = T)
str(h_data)
summary(h_data)
length(which(is.na(h_data)))
h_data <- na.omit(h_data)

# subset to 3-bed, 2- or 3-bath home
h_data <- subset(h_data,h_data$Bed==3 & h_data$Bath>=2 & h_data$Bath<=3)
h_data <- subset(h_data,select=-c(Bed,DateSold))
rows <- nrow(h_data)

# plot data
look1 <- h_data[,1:8]
look2 <- h_data[,c(1,9:ncol(h_data))]
pairs(look1)
pairs(look2)

split <- sample.int(n=rows,size=floor(.75*rows),replace=F)
training <- h_data[split, ]
test  <- h_data[-split, ]

# determine location clusters by k-means
clusterLoc <- kmeans(scale(training[,c(1,7,8)]),centers=20,iter.max=100,nstart=50)
clusterLoc$tot.withinss
clusterLoc
training$cluster <- factor(clusterLoc$cluster)
ggplot(data=training,aes(x=Long,y=Lat)) + geom_point(aes(color=cluster))
least <- aggregate(training$Price,list(training$cluster),mean)
least <- least[order(least$x),]

# fit support vector machines to predict location clusters by cv
nonsvm <- subset(training,select=c(cluster,Lat,Long,AreaHomeValue,AreaIncome))
tune.out <- tune(svm,factor(cluster)~.,data=nonsvm,kernel="radial",tunecontrol=tune.control(cross=5),
                 ranges=list(cost=c(0.1,1,10,100,1000), gamma=c(0.5,1,2,3,4)),scale=TRUE)
summary(tune.out)
tune.out$best.model
svmfit <- svm(factor(cluster)~.,data=nonsvm,kernel="radial",cost=1,gamma=1,scale=TRUE)
print(svmfit)

# change factors to dummy variables and drop the lowest one (to compare against)
training <- dummy.data.frame(training,sep=".")
head(least)
training <- subset(training,select=-cluster.10)

# fit regressions and determine by cross-validation
train.x <- as.matrix(subset(training,select=-Price))
train.y <- as.matrix(subset(training,select=Price))
train.model <- subset(training,select=-Price)
train.model["(Intercept)"] <- 1
train.model <- as.matrix(train.model)
rows.x <- nrow(train.x)
cols.x <- ncol(train.x)
folds <- sample(rep(1:5,length=rows.x))
table(folds)
cv.errors <- matrix(NA,5,cols.x)

# backwards stepwise
for(k in 1:5){
  best.fit <- regsubsets(x=train.x[folds!=k,],y=train.y[folds!=k],nvmax=cols.x,method="backward")
  for(i in 1:cols.x){
    coefi <- coef(best.fit,id=i)
    pred <- train.model[folds==k,names(coefi)] %*% coefi
    cv.errors[k,i] <- median(abs(train.y[folds==k,] - pred) / train.y[folds==k,])
  }
}
mape <- apply(cv.errors,2,mean)
plot(mape,pch=19,col="red",type="b")
mape
best.back <- regsubsets(x=train.x,y=train.y,nvmax=21,method="backward")

# forward stepwise
for(k in 1:5){
  best.fit <- regsubsets(x=train.x[folds!=k,],y=train.y[folds!=k],nvmax=cols.x,method="forward")
  for(i in 1:cols.x){
    coefi <- coef(best.fit,id=i)
    pred <- train.model[folds==k,names(coefi)] %*% coefi
    cv.errors[k,i] <- median(abs(train.y[folds==k] - pred) / train.y[folds==k])
  }
}
mape <- apply(cv.errors,2,mean)
plot(mape,pch=19,col="red",type="b")
mape
best.for <- regsubsets(x=train.x,y=train.y,nvmax=19,method="forward")

## compare the stepwise results to the following methods:

# fit ridge regression
fit.ridge <- glmnet(train.x,train.y,alpha=0)
plot(fit.ridge,xvar="lambda",label=T)
cv.ridge <- cv.glmnet(train.x,train.y,alpha=0)
plot(cv.ridge)
coef(cv.ridge)
coef.ridge <- as.vector(coef(cv.ridge))
names(coef.ridge) <- row.names(as.matrix(coef(cv.ridge)))

# fit lasso
fit.lasso <- glmnet(as.matrix(train.x),train.y,alpha=1)
plot(fit.lasso,xvar="lambda",label=T)
cv.lasso <- cv.glmnet(as.matrix(train.x),train.y,alpha=1)
plot(cv.lasso)
coef(cv.lasso)
coef.lasso <- as.vector(coef(cv.lasso))
names(coef.lasso) <- row.names(as.matrix(coef(cv.lasso)))

########################################################
# fit and predict sales for test data

# classifies test data into 20 locations
loc_test <- subset(test,select=c(Lat,Long,AreaHomeValue,AreaIncome))
best.mod <- tune.out$best.model
summary(best.mod)
test$cluster <- predict(best.mod,loc_test)

# dummy code factor variables (i.e., locations)
test$cluster <- as.factor(test$cluster)
test <- dummy.data.frame(test,sep=".")

## predict sale price for test data from 4 models
# (forward, backward, ridge, lasso)
test.model <- subset(test,select=-Price)
test.model["(Intercept)"] <- 1
predPrice <- as.data.frame(test$Price)
names(predPrice) <- "Price"

# forward stepwise
coefi <- coef(best.for,19)
predPrice$forward <- as.matrix(test.model[,names(coefi)])%*%coefi

# backward stepwise
coefi <- coef(best.back,21)
predPrice$backward <- as.matrix(test.model[,names(coefi)])%*%coefi

# ridge
predPrice$ridge <- as.matrix(test.model[,names(coef.ridge)])%*%coef.ridge

# lasso
predPrice$lasso <- as.matrix(test.model[,names(coef.lasso)])%*%coef.lasso

# final prediction (equally weighted average of 4 models)
predPrice$final <- 0.25*predPrice$forward + 0.25*predPrice$backward + 0.25*predPrice$ridge  + 0.25*predPrice$lasso
final.error <- apply(abs(((predPrice$Price - predPrice$final) / predPrice$Price* 100)),2,mean)
final.error
