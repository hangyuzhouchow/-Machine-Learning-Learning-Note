################## Note for machine learning in R #########################


##########Linear Model##############

###Validation Set Approach###
#We explore the use of the validation set approach in order to estimate the
#test error rates that result from fitting various linear models on the Auto
#data set.

library(ISLR)
set.seed(1)
train = sample(392,196)  ###Use sample function to split the the set of obs
lm.fit = lm(mpg~horsepower,data = Auto, subset = train)
attach(Auto)
mean((mpg-predict(lm.fit, Auto))[-train]^2)

library(boot)
glm.fit = glm(mpg~horsepower,data = Auto)
cv.err = cv.glm(Auto, glm.fit)
cv.err$delta

## LOOCV cv result in poly fits.
#We can use the poly() function to estimate the test error for the polynomial
#and cubic regressions.
cv.error = rep(0,5)
for (i in 1:5) {
  glm.fit = glm(mpg~poly(horsepower,i),data = Auto)
  cv.error[i] = cv.glm(Auto,glm.fit)$delta[1]
}
cv.error

## k-fold CV
set.seed(17)
cv.error.10 = rep(0,10)
for (i in 1:10) {
  glm.fit = glm(mpg~poly(horsepower,i),data = Auto)
  cv.error.10[i] = cv.glm(Auto,glm.fit, K = 10)$delta[1]
}
cv.error.10


###Bootstrap###
################Estimating the accuracy of a statistic interest #####
#1. create a function that computes the statistic of interest
#2. use boot() function to perform bootstrap by repeatedly 
# sampling obs from the data set with replacement.
alpha.fn =  function(data,index) {
  X = data$X[index]
  Y = data$Y[index]
  return((var(Y)-cov(X,Y))/(var(X)+var(Y)-2*cov(X,Y)))
}
# This function returns an estimate for alpha based on applying(5.7)
# to the observations indexed by the argument index.
alpha.fn(Portfolio,1:100)
set.seed(1)
alpha.fn(Portfolio,sample(100,100,replace = T))
boot(Portfolio,alpha.fn,R = 1000)


##########Estimating the accuracy of a linear regression model#####
boot.fn = function(data,index)
  return(coef(lm(mpg~horsepower,data = data, subset = index)))
boot.fn(Auto, 1:392)
## boot.fn can also be used in order to create bootstrap estimates
## for the intercept and slpe by randomly sampling from all obs with 
## replacement.
set.seed(1)
boot.fn(Auto,sample(392,392,replace = T))
boot.fn(Auto,sample(192,293,replace = T))
### Then use boot() to compute standard errors of 1000 bootstrap estimates
boot(Auto, boot.fn,1000)
#### Compare this to the results using standard formulas
summary(lm(mpg~horsepower,data = Auto))$coef


##########Fitting Classification Trees##########
library(tree)
attach(Carseats)
High = ifelse(Sales <= 8, "No", "Yes")
##use data.frame() to merge High with the data
Carseats = data.frame(Carseats, High)

tree.carseats = tree(High~. -Sales, Carseats)
summary(tree.carseats)
plot(tree.carseats)
text(tree.carseats,pretty = 0)
tree.carseats

### Calculate Test error
set.seed(2)
train = sample(1:nrow(Carseats),200)
Carseats.test = Carseats[-train,]
High.test = High[-train]
tree.carseats = tree(High~.-Sales, Carseats, subset = train)
tree.pred = predict(tree.carseats, Carseats.test, type="class")
table(tree.pred,High.test)
(86+57)/200

### prune the tree to see if we have improval
set.seed(3)
cv.carseats = cv.tree(tree.carseats, FUN = prune.misclass)
names(cv.carseats)
cv.carseats
par(mfrow = c(1,2))
plot(cv.carseats$size,cv.carseats$dev, type = "b")
plot(cv.carseats$k,cv.carseats$dev,type = "b")
prune.carseats = prune.misclass(tree.carseats, best = 9)
plot(prune.carseats)
text(prune.carseats,pretty = 0)
tree.pred = predict(prune.carseats, Carseats.test, type = "class")
table(tree.pred, High.test)
(94+60)/200


##### Fitting regression tree####
library(MASS)
set.seed(1)
train = sample(1:nrow(Boston),nrow(Boston)/2)
tree.boston = tree(medv~., Boston, subset = train)
summary(tree.boston)
plot(tree.boston)
text(tree.boston,pretty = 0)

cv.boston = cv.tree(tree.boston)
plot(cv.boston$size, cv.boston$dev, type = 'b')
prune.boston = prune.tree(tree.boston,best = 5)
plot(prune.boston)
text(prune.boston,pretty= 0)
yhat = predict(tree.boston, newdata = Boston[-train,])
boston.test = Boston[-train,"medv"]
plot(yhat, boston.test)
abline(0,1)
mean((yhat-boston.test)^2)

##### Boosting####
library(gbm)
set.seed(1)
boost.boston = gbm(medv~., data = Boston[train,],distribution = "gaussian",
                   n.trees = 5000, interaction.depth =4)
summary(boost.boston)
par(mfrow = c(1,2))
plot(boost.boston,i ="rm")
plot(boost.boston,i = "lstat")
yhat.boost = predict(boost.boston,newdata = Boston[-train,],n.trees = 5000)
mean((yhat.boost-boston.test)^2)
## change shrinkage λ
boost.boston = gbm(medv~., data = Boston[train,],distribution = "gaussian",
                   n.trees = 5000, interaction.depth =4, shrinkage = 0.2)
yhat.boost = predict(boost.boston,newdata = Boston[-train,],n.trees = 5000)
mean((yhat.boost-boston.test)^2)



##### Best Subset Selection#####
library(ISLR)
names(Hitters)
dim(Hitters)
sum(is.na(Hitters$Salary))
Hitters = na.omit(Hitters)
dim(Hitters)

library(leaps)
regfit.full = regsubsets(Salary~.,Hitters)
summary(regfit.full)
names(summary)
### by default, we only have 8 best variables, we can reset using nvmax
regfit.full = regsubsets(Salary~.,Hitters, nvmax = 19)
reg.summary = summary(regfit.full)
names(reg.summary)
reg.summary$rsq
par(mfrow = c(2,2))
plot(reg.summary$rss,xlab = "Number of Variables", ylab = "RSS",type = "l")
plot(reg.summary$adjr2,xlab = "Number of Variables", ylab = "Adjusted RSq",type = "l")
which.max(reg.summary$adjr2)
points(11,reg.summary$adjr2[11],col = 'red',cex = 2, pch = 20)
plot(reg.summary$cp, xlab = "Number of Variables", ylab = "Cp",type = "l")
which.min(reg.summary$cp)
points(10,reg.summary$cp[10], col = "red", cex = 2, pch = 20)
which.min(reg.summary$bic)
plot(reg.summary$bic, xlab = "Number of Variables", ylab = "BIC",type = "l")
points(6,reg.summary$bic[6],col = "red", cex = 2, pch = 20)

### regsubsets() function has a built-in plot() which can be used to
### display the selected variables for the best model with a given number
### of predictors, ranked according to the BIC, Cp, adjusted R2, or AIC.
plot(regfit.full,scale = "r2")
plot(regfit.full,scale = "adjr2")
plot(regfit.full,scale = "Cp")
plot(regfit.full,scale = "bic")

######## Forward and Backward Stepwise Selection #######
regfit.fwd = regsubsets(Salary~., data = Hitters, nvmax = 19, method = "forward")
regfit.bwd = regsubsets(Salary~., data = Hitters, nvmax = 19, method = "backward")
summary(regfit.fwd)
summary(regfit.bwd)
coef(regfit.full,7)
coef(regfit.fwd,7)
coef(regfit.bwd,7)

##### Choosing Among Models Using the Validation set Approach & CV
set.seed(1)
train = sample(c(TRUE, FALSE), nrow(Hitters),rep = TRUE)
test = (!train)
regfit.best = regsubsets(Salary~., data = Hitters[train,], nvmax = 19)
test.mat = model.matrix(Salary~., data = Hitters[test,])
## model.matrix() function is used to buil an "X" matrix from data
val.errors = rep(NA,19)
for (i in 1:19) {
  coefi = coef(regfit.best, id = i)
  pred = test.mat[,names(coefi)]%*%coefi
  val.errors[i] = mean((Hitters$Salary[test]-pred)^2)
}
val.errors
which.min(val.errors)
coef(regfit.best, 10)

predict.regsubsets = function(object, newdata,id,...){
  form = as.formula(object$call[[2]])
  mat = model.matrix(form,newdata)
  coefi = coef(object, id = id)
  xvars = names(coefi)
  mat[,xvars] %*% coefi
}
regfit.best = regsubsets(Salary~., data = Hitters, nvmax = 19)
coef(regfit.best,10)

######### cross-validation #####
k = 10
set.seed(1)
folds = sample(1:k, nrow(Hitters), replace = T)
cv.errors = matrix(NA, k, 19, dimnames = list(NULL, paste(1:19)))
for (j in 1:k){
  best.fit = regsubsets(Salary~., data = Hitters[folds!=j,],
                        nvmax = 19)
  for (i in 1:19){
    pred = predict(best.fit, Hitters[folds ==j, ],id = i)
    cv.errors[j,i] = mean((Hitters$Salary[folds==j]-pred)^2)
  }
}
mean.cv.errors = apply(cv.errors,2,mean)
mean.cv.errors
par(mfrow = c(1,1))
plot(mean.cv.errors,type = "b")
reg.best = regsubsets(Salary~., data = Hitters, nvmax = 19)
coef(reg.best,11)


######## Ridge regression and the Lasso
library(glmnet)
x = model.matrix(Salary~.,Hitters)[,-1]
y = Hitters$Salary
grid = 10^seq(10,-2,length = 100)
### if alpha =0, then ridge regression applys, alpha = 1, then lasso applys
### by default, glmnet() standardized the variables, use
### standardize = FALSE to turn off the settings.
ridge.mod = glmnet(x,y,alpha = 0, lambda = grid)
## to get lambda, use coef()
dim(coef(ridge.mod))
ridge.mod$lambda[50]
coef(ridge.mod)[,50]
sqrt(sum(coef(ridge.mod)[-1,50]^2))
predict(ridge.mod, s=50, type = "coefficients")[1:20,]

#### apply ridge and lasso using random subset
set.seed(1)
train = sample(1:nrow(x),nrow(x)/2)
test = (-train)
y.test = y[test]
ridge.mod = glmnet(x[train,],y[train], alpha = 0, lambda = grid, 
                   thresh = 1e-12)
ridge.pred = predict(ridge.mod, s = 4, newx = x[test,])
mean((ridge.pred - y.test)^2)
mean((mean(y[train])-y.test)^2)


ridge.pred = predict(ridge.mod, s = 1e10, newx = x[test,])
mean((ridge.pred-y.test)^2)


ridge.pred = predict(ridge.mod, s = 0, newx = x[test,])
mean((ridge.pred-y.test)^2)
lm(y~x, subset = train)
predict(ridge.mod,s=0, exact = T, type= "coefficients")[1:20.]

### in general, it's better to use cv to choose lambda.
set.seed(1)
cv.out = cv.glmnet(x[train,],y[train],alpha = 0)
plot(cv.out)
bestlam = cv.out$lambda.min
bestlam
ridge.pred = predict(ridge.mod, s=bestlam, newx = x[test,])
mean((ridge.pred-y.test)^2)
out = glmnet(x,y,alpha = 0)
predict(out, type = "coefficients", s = bestlam)[1:20,]

### PCR
library(pls)
set.seed(2)
pcr.fit = pcr(Salary~., data = Hitters, scale = TRUE, validation = "CV")
summary(pcr.fit)
validationplot(pcr.fit, val.type = "MSEP")
