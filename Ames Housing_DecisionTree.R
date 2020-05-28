library(caTools)    # data splitting 
library(dplyr)       # data wrangling
library(rpart)       # performing regression trees
library(rpart.plot)  # plotting regression trees
library(rattle)
library(AmesHousing)
library(ipred)

df <- AmesHousing::make_ames()
summary(df)


#split data
set.seed(101)
sample = sample.split(df$MS_Zoning, SplitRatio = 0.7)
df.train <- subset(df, sample == TRUE)
df.test <- subset(df, sample == FALSE)

#create tree model
m1 <- rpart(Sale_Price~., data = df.train, method = 'anova')

#display results text
m1

#plot tree rpart automatically applies cost fucntion (10fold CV to compute error)
rpart.plot(m1)
fancyRpartPlot(m1)
prp(m1, varlen=5)

#plot error based on size of tree / y-axis is CV error / lower x is cost complexity
#rpart trimmed to 12 nodes
plotcp(m1)

#results / xerror = CVE .259 @ 12 nodes
m1$cptable



####################Tuning with control

#w/o triming / cp=0 no error penalty
#m1.1 <- rpart(Sale_Price~., data = df.train, method = 'anova', control=list(cp=0, xval=10))
#plotcp(m1.1)
#abline(v=12, lty ='dashed')


#minsplit = min # of data points req. for split before creating terminal node
#maxdepth = max number of internal nodes between root and and terminal nodes (default 30)
#improves cv error to .227
m2 <- rpart(Sale_Price~., data = df.train, method = 'anova', control=list(minsplit = 10, maxdepth = 12, xval=10))
m2$cptable 

#create grid to optimize minsplit and maxdepth instead of testing multiple models
#maxdepth 8-15 since previous optimum was 12
grid <- expand.grid(
  minsplit = seq(5, 20, 1),
  maxdepth = seq(8, 15, 1)
  )
head(grid)

#create loop to iterate each minsplit and maxdepth
models <- list()
for (i in 1:nrow(grid)) {
  #minsplit maxdepth values at i
  minsplit <- grid$minsplit[i]
  maxdepth <- grid$maxdepth[i]
  
  #train model and store in the list
  models[[i]] <- rpart(Sale_Price~.,
                       data = df.train,
                       method = 'anova',
                       control = list(minsplit = minsplit, maxdepth = maxdepth)
                       
        )
}


#function for optimal cp
get_cp <- function(x) {
  min <- which.min(x$cptable[, 'xerror'])
  cp <- x$cptable[min, 'CP']
}

#function for minimum error
get_min_error <- function(x) {
  min <-which.min(x$cptable[, 'xerror'])
  xerror <- x$cptable[min, 'xerror']
}

grid %>%
  mutate(
    cp = purrr::map_dbl(models, get_cp),
    error = purrr::map_dbl(models, get_min_error)
  ) %>%
  arrange(error) %>%
  top_n(-5, wt = error)    


#optimal tree with minsplit: 7 & maxdepth: 8
optimal_tree <- rpart(
  Sale_Price~.,
  data = df.train,
  method = 'anova',
  control = list(minsplit = 7, maxdepth = 8, cp = .01)
)

optimal_tree
fancyRpartPlot(optimal_tree)
optimal_tree$cptable

#rmse
library('Metrics')
pred <- predict(optimal_tree, newdata = df.test)
rmse(pred,df.test$Sale_Price)







######################################################################################
#bagging trees
  
  #split data
  set.seed(101)
sample = sample.split(df$MS_Zoning, SplitRatio = 0.7)
df.train <- subset(df, sample == TRUE)
df.test <- subset(df, sample == FALSE)

#bootstrapping
set.seed(123)


bagged_m1 <- bagging(
  Sale_Price~.,
  data = df.train,
  coob = TRUE
)
bagged_m1


#asses 10-50 bagged trees
ntree <- 10:60

#create empty vector to store OOB RMSE values
rmse <- vector(mode = 'numeric', length = length(ntree))

for (i in seq_along(ntree)) {
  set.seed(123)
  
  #model
  model <- bagging(
    Sale_Price~.,
    data = df.train,
    coob = TRUE,
    nbagg = ntree[i]
  )
  
  #get OOB error
  rmse[i] <- model$err
}

plot(ntree, rmse, type = 'l', lwd = 2)
abline(v = 50, col = "red", lty = "dashed")


#bootstrapping tuned model
set.seed(123)


bagged_tuned <- bagging(
  Sale_Price~.,
  data = df.train,
  coob = TRUE,
  nbagg = 50
)
bagged_tuned


#rmse
library('Metrics')
pred <- predict(bagged_tuned, newdata = df.test)
rmse(pred,df.test$Sale_Price)