Practical Machine Learning Assignment 1
========================================================
### Task : The goal is to predict the manner in which the users did the exercise.

At first we will load the necessary packages

```{r}
library(caret)
library(ggplot2)
library(randomforest)
library(rpart.plot)
```
### Getting the Data

Read the **training data** and the **testing data** from the url provided

```{r}
urltrain<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urltest<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training<-read.csv(url(urltrain),na.strings=c("#DIV/0!"))
testing<-read.csv(url(urltest),na.strings=c("#DIV/0!"))
```

This is the training data that has 19622 observations and 160 predictors

### Basic Preprocessing

Remove the columns that have all Na's

```{r}
newtrain<-training[,which(colSums(is.na(training)) == 0)]
dim(newtrain)
newtest<-testing[,which(colSums(is.na(testing)) == 0)]
dim(newtest)
```
We now have the new training and test sets i.e **newtrain** and **newtest** that do not have NA's.

The new training data has 19622 observations and 127 predictors. 

The new test data has 20 observations and 127 predictors.

We will now filter out the less important features like the *user_name* as name cannot be a good predictor, *X* as it is the observation id.   

We also remove al the predictors having the keyword *timestamp* or *window*    

We also force the columns into numeric values.     

We will do this for the training as well as the test data sets. 


```{r}
finalclass<-newtrain$classe
trainfilter<-grep("^X|timestamp|window|user_name",names(newtrain))
newtrain<-newtrain[, -trainfilter]
cleantrain<- newtrain[, sapply(newtrain, is.numeric)]
cleantrain$classe <- classe
dim(cleantrain)
```
```{r}
testfilter<-grep("^X|timestamp|window|user_name",names(newtest))
newtest<-newtest[, -testfilter]
dim(newtest)
cleantest<- newtest[, sapply(newtest, is.numeric)]
dim(cleantest)
```

This *cleantrain* data set now have 19622 observations and 53 predictors.

The *cleantest* data set has 20 observations and 53 predictors.

### Data Slicing

We will now devide the *cleantest* data set into **training data** and **validation data**.    

```{r}
set.seed(12345)
inTrain<-createDataPartition(cleantrain$classe,p=0.7,list=FALSE)
traindata<-cleantrain[inTrain,]
valdata<-cleantrain[-inTrain,]
```

### Fit the Model 

We will set the training control parameters for a cross validation of 5.   

```{r}
fitControl <- trainControl(method = "cv", 5)
```
We will now train the *traindata* using the **random-forest** algorithm.

We use *Random Forest* as unlike single decision trees which are likely to suffer from 
high variance the *random forest* algorithm use averaging to find a natural balance between
the two extremes. It automatically selects important variables and is robust to correlated covariates & outliers.   

```{r}
set.seed(12876)
modfit <- train(classe ~ ., data = traindata, method = "rf",trControl=fitControl,ntree=250)
modfit
```

```{r}
## Random Forest 

## 13737 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 

## No pre-processing
## Resampling: Cross-Validated (5 fold) 

## Summary of sample sizes: 10990, 10991, 10989, 10989, 10989 

## Resampling results across tuning parameters:

##  mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2    0.990     0.988  0.00222      0.00281 
##  27    0.990     0.988  0.00277      0.00350 
##  52    0.985     0.981  0.00369      0.00467 

## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

### Performance Evaluation

We evaluate the performance on the *validation data set*    

```{r}
pred1<-predict(modfit,valdata)
```

We build the **confusion matrix** 

```{r}
confusionMatrix(pred1,valdata$classe)
```

```{r}
##Confusion Matrix and Statistics

##          Reference
## Prediction    A    B    C    D    E
##         A 1672   11    0    0    0
##         B    2 1122   14    0    0
##         C    0    6 1008   25    1
##         D    0    0    4  939    4
##         E    0    0    0    0 1077

## Overall Statistics
                                          
##               Accuracy : 0.9886          
##                 95% CI : (0.9856, 0.9912)
##    No Information Rate : 0.2845          
##    P-Value [Acc > NIR] : < 2.2e-16       
                                          
##                  Kappa : 0.9856          
## Mcnemar's Test P-Value : NA              

## Statistics by Class:

##                     Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9851   0.9825   0.9741   0.9954
## Specificity            0.9974   0.9966   0.9934   0.9984   1.0000
## Pos Pred Value         0.9935   0.9859   0.9692   0.9916   1.0000
## Neg Pred Value         0.9995   0.9964   0.9963   0.9949   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1907   0.1713   0.1596   0.1830
## Detection Prevalence   0.2860   0.1934   0.1767   0.1609   0.1830
## Balanced Accuracy      0.9981   0.9909   0.9879   0.9862   0.9977
```

Here, the **Accuracy = 0.9886151**     
and the **Kappa = 0.9855962**    

We will now calculate the *out of sample error*    

```{r}
error<-1 - (confusionMatrix(pred1,valdata$classe))$overall[[1]]
error
```
The out of sample error is  **0.01138488**     

### Predicting the test data set

```{r}
pred2<-predict(modfit, cleantest)
pred2
```

```{r}
## [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

### Visulaization

```{r}
prp(rpart(classe~., data = valdata, method="class"))
```

![alt text][id]
[id]: Rplot01.png 

