# Loading data and import libraries
loan_data_test<-read.csv('loan_data_test.csv', header = T)
loan_data_test <- loan_data_test[,-1]
pacman::p_load(Rcpp,Amelia,magrittr,dplyr,neuralnet,NeuralNetTools,SPOT,class,gmodels,
               e1071,lattice,ggplot2,caret,tictoc,randomForest,tidyverse,rpart)

# prediction model:Neural Network, Linear Regression  

## Split data into train and test set
indx <- sample(1:nrow(loan_data_test), as.integer(0.75*nrow(loan_data_test)))
loan_data_test_train <- loan_data_test[indx,]
loan_data_test_test <- loan_data_test[-indx,]
loan_data_test_train_label <- loan_data_test_train[,1]
loan_data_test_test_label <- loan_data_test_test[,1]

## Neural Network
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
data_b_norm <- as.data.frame(lapply(loan_data_test[,-1],normalize))
summary(data_b_norm$loan_status)
data_b_train <- loan_data_test[indx,]
data_b_test <- loan_data_test[-indx,]
set.seed(999)
data_b_model <- neuralnet(formula = loan_status ~ annual_inc + installment + dti + funded_amnt_inv 
                          +fico_range_low + fico_range_low + revol_bal + inq_last_6mths + 
                            loan_amnt + open_acc + term + home_ownership + grade + emp_length,
                          data = data_b_train)
plot(data_b_model)
plotnet(data_b_model,alpha = 0.6)
NNmodel_result <- compute(data_b_model,data_b_test)
NNpred <- NNmodel_result$net.result
MSE <- function(actual, predicted) {
  mean((actual - predicted)^2)  }
MSE(NNpred, data_b_test$loan_status) # MSE = 0.1378


## Linear Regression

data_LR_train <- loan_data_test[indx,]
data_LR_test <- loan_data_test[-indx,]
LRmodel <- lm(formula = loan_status ~ annual_inc + installment + dti + funded_amnt_inv 
              +fico_range_low + fico_range_low + revol_bal + inq_last_6mths + 
                loan_amnt + open_acc + term + home_ownership + grade + emp_length,
              data = data_LR_train)
summary(LRmodel)
LR_pred <- predict(LRmodel, data_LR_test)
summary(LR_pred)
summary(data_LR_test$loan_status)
plot(LR_pred,data_LR_test$loan_status)
cor(LR_pred,data_LR_test$loan_status)
MSE <- function(actual, predicted) {
  mean((actual - predicted)^2)  }
MSE(LR_pred, data_LR_test$loan_status) # MSE = 0.1317

# classification model: Logistics Regression, KNN, Random Forest, Naive Bayes

## Logistics Regression

LRmodel <- glm(loan_status ~.,family=binomial(link='logit'), data = loan_data_test_train)
summary(LRmodel)

LRmodel2 <- glm(loan_status ~annual_inc + installment + dti + funded_amnt_inv + 
                  fico_range_low + revol_bal + inq_last_6mths + loan_amnt + open_acc + 
                  term + home_ownership + grade + emp_length,family=binomial(link='logit'),
                data = loan_data_test_train) # take off fico_range_high
summary(LRmodel2)
# trainning
LRfitted.results <- predict(LRmodel2, newdata = loan_data_test_train, type = 'response')
head(LRfitted.results)
head(loan_data_test_train_label)
LRfitted.results <- ifelse(LRfitted.results > 0.7,"Fully Paid","Charged Off")
head(LRfitted.results)
misClasificError <- mean(LRfitted.results != loan_data_test_train$loan_status, na.rm=TRUE)
print(paste('Accuracy',1-misClasificError))

# test
LRfitted.results <- predict(LRmodel2, newdata = loan_data_test_test, type = 'response')
head(LRfitted.results)
LRfitted.results <- ifelse(LRfitted.results > 0.7,"Fully Paid","Charged Off")
head(LRfitted.results)
head(loan_data_test_test_label)
misClasificError <- mean(LRfitted.results != loan_data_test_test$loan_status, na.rm=TRUE)
print(paste('Accuracy',1-misClasificError))

confusionMatrix(data = as.factor(LRfitted.results), as.factor(loan_data_test_test_label))
CrossTable(LRfitted.results,loan_data_test_test_label,prop.chisq = FALSE,
           prop.t = FALSE, prop.r = FALSE,dnn = c('predicted', 'actual'))
LRmodel3 <- glm(loan_status ~annual_inc + dti + fico_range_low + inq_last_6mths + 
                  term + home_ownership + grade + emp_length,family=binomial(link='logit'),
                data = loan_data_test_train) # take off fico_range_high
summary(LRmodel3)

# test
LRfitted.results <- predict(LRmodel3, newdata = loan_data_test_test, type = 'response')
head(LRfitted.results)
LRfitted.results <- ifelse(LRfitted.results > 0.7,"Fully Paid","Charged Off")
head(LRfitted.results)
head(loan_data_test_test_label)
misClasificError <- mean(LRfitted.results != loan_data_test_test$loan_status, na.rm=TRUE)
print(paste('Accuracy',1-misClasificError))

confusionMatrix(data = as.factor(LRfitted.results), as.factor(loan_data_test_test_label))
CrossTable(LRfitted.results,loan_data_test_test_label,prop.chisq = FALSE,
           prop.t = FALSE, prop.r = FALSE,dnn = c('predicted', 'actual'))
# Accuracy of Logistics Regression model : 0.7944

## K Nearest Neighbors(KNN)

set.seed(100)
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
loan_data_test_n <- as.data.frame(lapply(loan_data_test[,-1],normalize))
indx <- sample(1:nrow(loan_data_test_n), as.integer(0.75*nrow(loan_data_test_n)))
loantestdata_train <- loan_data_test_n[indx,]
loantestdata_test <- loan_data_test_n[-indx,]
loantestdata_train_label <- loan_data_test[indx,1]
loantestdata_test_label <- loan_data_test[-indx,1]
prednn <- knn(train = loantestdata_train, test = loantestdata_test,cl = loantestdata_train_label, k = 42)
head(prednn)
cm <- table(x = loantestdata_test_label, y = prednn, dnn = c("actual", "predict"))
CrossTable(x = loantestdata_test_label, y = prednn,prop.chisq = FALSE)
misCEknn <- mean(prednn !=loantestdata_test_label, na.rm = TRUE)
print(paste('Accuracy',1-misCEknn))
# Accuracy of KNN model : 0.82715
  
  
## Random Forest 

dtrain_rf <- loan_data_test[indx,]
dtest_rf <- loan_data_test[-indx,]
tic()
rf_c <- randomForest(loan_status~.,data = dtrain_rf)
toc()
rf_c
summary(rf_c)
rf_pred <- predict(rf_c, dtest_rf)
confusionMatrix(data = rf_pred, as.factor(dtest_rf$loan_status))
# Accuracy of Random Forest model : 0.8242
  
  
## Naive Bayes

dtrain_nb <- loan_data_test[indx,]
dtest_nb <- loan_data_test[-indx,]
nbcm_b <- naiveBayes(dtrain_nb[,-1],dtrain_nb$loan_status)
prednb1 <- predict(nbcm_b,dtest_nb[,-1])
CrossTable(prednb1,dtest_nb$loan_status,prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,dnn = c('predicted', 'actual'))
misCEnb <- mean(prednb1 !=dtrain_nb$loan_status, na.rm = TRUE)
print(paste('Accuracy',1-misCEnb))
# Accuracy of Naive Bayes model : 0.71224

  
