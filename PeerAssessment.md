# Project Report - Practical Machine Learning - Prediction in Qualitative Activity Recognition of Weight Lifting Exercises

### Executive Summary

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify <i>how well they do it</i>. In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.  
The objective of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set.  
To do so, we used the data available to build a training and a testing data set in order to compute a prediction model and to validate it. Those data contain a lot of variables, more or less correlated to each other, so we needed first to proceed with cleaning actions to keep only the relevant informations.  
Our analysis led to say that the random forest predictor is much more accurate than the CART model. And random forest predictor is good enough to be used as a predictor for any input variables since the uncertainty rate is very low (99.8% of accuracy based on the testing data set).  

NB: For more details regarding the data set, please refer to appendix 1.  



After loading the data set from the files 'pml-training.csv' and 'pml-testing.csv' available on the Website 'cloudfront.net', some quick exploratory actions have been performed and moved to appendix 2.  


```r
training_url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testing_url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
# Load the data sets from given URL and consider NA every empty values or values equal to '#DIV/0!'
training <- read.csv(url(training_url), na.strings = c("NA","#DIV/0!",""))
testing <- read.csv(url(testing_url), na.strings = c("NA","#DIV/0!",""))
```

***
### Data transformation

We will first split the csv file 'training' and build our 'training' and 'testing' data sets for our analysis. Then, we will reduce the number of variables from 160 to 58 (including our outcome 'classe') by removing those which are computations of others (and consequently very corrolated to others).  


```r
set.seed(12345)
# Partitioning the training data in 2: training with 70% of the original dataset
# and testing with the remaining 30%
inTrain <- createDataPartition(y = training$classe, p = 0.7, list = FALSE)
train_data <- training[inTrain, ]
test_data <- training[-inTrain, ]

# Data transformation for further purpose
# Factorize the outcome 'classe' and variable 'user_name' which is a character string
# (this might be a problem with randomforest if not factorized)
train_data$classe <- as.factor(train_data$classe)
train_data$user_name <- as.factor(train_data$user_name)

test_data$classe <- as.factor(test_data$classe)
test_data$user_name <- as.factor(test_data$user_name)

# The testing data set does not contain the outcome 'classe'
testing$user_name <- as.factor(testing$user_name)

# Remove useless variables
# Dimensions of our data sets before cleaning
dim(train_data)
```

```
## [1] 13737   160
```

```r
dim(test_data)
```

```
## [1] 5885  160
```

```r
dim(testing)
```

```
## [1]  20 160
```

```r
# We keep only variables related to measurements and we get rid of all the columns which
# are the results of computation of the others, and so, are very correlated to some others
pattern <- "^(X|new_|kurtosis_|skewness_|max_|min_|amplitude_|avg_|stddev_|var_|skewness_$)"
columns_to_drop <- grep(pattern = pattern, names(train_data), value = TRUE)
# Number of columns to drop
print(length(columns_to_drop))
```

```
## [1] 102
```

```r
columns_to_keep <- names(train_data)[!(names(train_data) %in% columns_to_drop)]
train_data <- train_data[, columns_to_keep]
test_data <- test_data[, columns_to_keep]
# The testing data set does not contain the outcome 'classe'
columns_to_keep <- columns_to_keep[-58]
testing <- testing[, columns_to_keep]

# Dimensions of our data sets after cleaning
dim(train_data)
```

```
## [1] 13737    58
```

```r
dim(test_data)
```

```
## [1] 5885   58
```

```r
dim(testing)
```

```
## [1] 20 57
```

***
### Analysis

Now that the data sets are ready, we can proceed with our analysis.  
First we will fit a CART model with the rpart method using all predictor variables we kept and method 'class'. And we use our testing data set 'test_data' to check the accuracy of our model.  


```r
# CART model using rpart method
modCART <- rpart(classe ~ ., data = train_data, method = "class")
# We can plot the resulting tree using the function 'fancyRpartPlot'
fancyRpartPlot(modCART)
```

![](PeerAssessment_files/figure-html/model_selection_rpart-1.png)<!-- -->

```r
predictCART <- predict(modCART, test_data, type = "class")
table(test_data$classe)
```

```
## 
##    A    B    C    D    E 
## 1674 1139 1026  964 1082
```

```r
table(predictCART)
```

```
## predictCART
##    A    B    C    D    E 
## 1684 1086 1296  872  947
```

```r
confusionMatrix(predictCART, test_data$classe)$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.8723874      0.8385952      0.8635929      0.8808109      0.2844520 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```

Then, we fit a random forest predictor relating the factor variable 'classe' to the remaining variables. And again, we use our testing data set 'test_data' to check the accuracy of our new model.  


```r
# Random forest predictor
modRF <- randomForest(classe ~ ., data = train_data)
predictRF <- predict(modRF, test_data, type = "class")
table(test_data$classe)
```

```
## 
##    A    B    C    D    E 
## 1674 1139 1026  964 1082
```

```r
table(predictRF)
```

```
## predictRF
##    A    B    C    D    E 
## 1673 1142 1022  968 1080
```

```r
confusionMatrix(predictRF, test_data$classe)$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9988105      0.9984955      0.9975508      0.9995216      0.2844520 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```

From here, we can see that the <span style="color:red;background-color:#eeeeee;padding:3px;border-radius:3px 3px 3px;border:.5px solid lightgrey">random forest predictor has a far better accuracy of ``99.88`` %</span> when the <span style="color:red;background-color:#eeeeee;padding:3px;border-radius:3px 3px 3px;border:.5px solid lightgrey">CART model meets only ``87.24`` % of the expected outcomes</span>. We will preferably use the random forest for any prediction rather than the CART model.  
<span style="color:red;background-color:#eeeeee;padding:3px;border-radius:3px 3px 3px;border:.5px solid lightgrey">The uncertainty is very low and expercted errors from outcome can be considered as null when using random forest in our case.</span>  

***
### Prediction on data set Testing

Finally, we use our random forest model to predict the outcome of our 'testing' data set with 20 different test cases.  


```r
# The random forest predictor is very tricky with the class types of the variables.
# So we need to be sure that variables of our 'testing' data set are with the exact
# same class types of the data set 'train_data' that was used to build the
# random forest predictor.
# Below is a quick way to coerce the variables of 'testing' to class types from 'train_data':
class_testing <- test_data[1,-58]
testing <- rbind(class_testing, testing[1:20,])
testing <- testing[2:21,]
rownames(testing) <- 1:nrow(testing)

# Prediction of 'testing' outcome using random forest model
resultRF <- predict(modRF, testing, type = "class")
resultRF
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

For comparison, below is the prediction using the CART model.  


```r
# Prediction of 'testing' outcome using rpart model
resultCART <-  predict(modCART, testing, type = "class")
resultCART
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  C  A  A  E  D  C  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

***
### Appendix

1. A few details regarding our data set  

The outcome **classe**: different fashions that can take the set of 10 repetitions of the Unilateral Dumbbell Biceps Curl:  
    . A: exactly according to the specification  
    . B: throwing the elbows to the front  
    . C: lifting the dumbbell only halfway  
    . D: lowering the dumbbell only halfway  
    . E: throwing the hips to the front  

<div class = "row">
<div class = "col-sm-4">

![](./pictures/on-body-sensing-schema.png){width=260px}

</div>
<div class = "col-md-8">

This human activity recognition research has traditionally focused on discriminating between different activities, i.e. to predict "which" activity was performed at a specific point in time (like with the Daily Living Activities dataset above). The approach we propose for the Weight Lifting Exercises dataset is to investigate "how (well)" an activity was performed by the wearer. The "how (well)" investigation has only received little attention so far, even though it potentially provides useful information for a large variety of applications,such as sports training.  

In this work (see the paper) we first define quality of execution and investigate three aspects that pertain to qualitative activity recognition: the problem of specifying correct execution, the automatic and robust detection of execution mistakes, and how to provide feedback on the quality of execution to the user. We tried out an on-body sensing approach (dataset here), but also an "ambient sensing approach" (by using Microsoft Kinect - dataset still unavailable)  

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).  

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).  

</div>
</div>

***
### Credit
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.  

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz4tBCtG1oY  

2. Exploratory data analysis  

```r
dim(train_data)
```

```
## [1] 13737    58
```

```r
names(train_data)
```

```
##  [1] "user_name"            "raw_timestamp_part_1" "raw_timestamp_part_2"
##  [4] "cvtd_timestamp"       "num_window"           "roll_belt"           
##  [7] "pitch_belt"           "yaw_belt"             "total_accel_belt"    
## [10] "gyros_belt_x"         "gyros_belt_y"         "gyros_belt_z"        
## [13] "accel_belt_x"         "accel_belt_y"         "accel_belt_z"        
## [16] "magnet_belt_x"        "magnet_belt_y"        "magnet_belt_z"       
## [19] "roll_arm"             "pitch_arm"            "yaw_arm"             
## [22] "total_accel_arm"      "gyros_arm_x"          "gyros_arm_y"         
## [25] "gyros_arm_z"          "accel_arm_x"          "accel_arm_y"         
## [28] "accel_arm_z"          "magnet_arm_x"         "magnet_arm_y"        
## [31] "magnet_arm_z"         "roll_dumbbell"        "pitch_dumbbell"      
## [34] "yaw_dumbbell"         "total_accel_dumbbell" "gyros_dumbbell_x"    
## [37] "gyros_dumbbell_y"     "gyros_dumbbell_z"     "accel_dumbbell_x"    
## [40] "accel_dumbbell_y"     "accel_dumbbell_z"     "magnet_dumbbell_x"   
## [43] "magnet_dumbbell_y"    "magnet_dumbbell_z"    "roll_forearm"        
## [46] "pitch_forearm"        "yaw_forearm"          "total_accel_forearm" 
## [49] "gyros_forearm_x"      "gyros_forearm_y"      "gyros_forearm_z"     
## [52] "accel_forearm_x"      "accel_forearm_y"      "accel_forearm_z"     
## [55] "magnet_forearm_x"     "magnet_forearm_y"     "magnet_forearm_z"    
## [58] "classe"
```

```r
summary(train_data)
```

```
##     user_name    raw_timestamp_part_1 raw_timestamp_part_2
##  adelmo  :2749   Min.   :1.322e+09    Min.   :   294      
##  carlitos:2199   1st Qu.:1.323e+09    1st Qu.:254958      
##  charles :2418   Median :1.323e+09    Median :500340      
##  eurico  :2122   Mean   :1.323e+09    Mean   :502510      
##  jeremy  :2386   3rd Qu.:1.323e+09    3rd Qu.:752385      
##  pedro   :1863   Max.   :1.323e+09    Max.   :998801      
##                                                           
##           cvtd_timestamp   num_window      roll_belt     
##  28/11/2011 14:14:1033   Min.   :  1.0   Min.   :-28.90  
##  05/12/2011 11:24:1028   1st Qu.:220.0   1st Qu.:  1.10  
##  05/12/2011 11:25:1026   Median :424.0   Median :113.00  
##  30/11/2011 17:11:1014   Mean   :430.1   Mean   : 64.44  
##  02/12/2011 13:34: 985   3rd Qu.:643.0   3rd Qu.:123.00  
##  05/12/2011 14:23: 965   Max.   :864.0   Max.   :162.00  
##  (Other)         :7686                                   
##    pitch_belt         yaw_belt       total_accel_belt  gyros_belt_x      
##  Min.   :-54.700   Min.   :-180.00   Min.   : 0.00    Min.   :-1.040000  
##  1st Qu.:  1.690   1st Qu.: -88.30   1st Qu.: 3.00    1st Qu.:-0.050000  
##  Median :  5.280   Median : -13.00   Median :17.00    Median : 0.030000  
##  Mean   :  0.231   Mean   : -10.91   Mean   :11.32    Mean   :-0.007106  
##  3rd Qu.: 14.800   3rd Qu.:  13.60   3rd Qu.:18.00    3rd Qu.: 0.110000  
##  Max.   : 60.200   Max.   : 179.00   Max.   :28.00    Max.   : 2.220000  
##                                                                          
##   gyros_belt_y       gyros_belt_z      accel_belt_x       accel_belt_y   
##  Min.   :-0.51000   Min.   :-1.3500   Min.   :-120.000   Min.   :-69.00  
##  1st Qu.: 0.00000   1st Qu.:-0.2000   1st Qu.: -21.000   1st Qu.:  3.00  
##  Median : 0.02000   Median :-0.1000   Median : -15.000   Median : 35.00  
##  Mean   : 0.03958   Mean   :-0.1311   Mean   :  -5.506   Mean   : 30.15  
##  3rd Qu.: 0.11000   3rd Qu.:-0.0200   3rd Qu.:  -5.000   3rd Qu.: 61.00  
##  Max.   : 0.64000   Max.   : 1.6100   Max.   :  85.000   Max.   :164.00  
##                                                                          
##   accel_belt_z     magnet_belt_x    magnet_belt_y   magnet_belt_z   
##  Min.   :-269.00   Min.   :-48.00   Min.   :359.0   Min.   :-623.0  
##  1st Qu.:-162.00   1st Qu.:  9.00   1st Qu.:582.0   1st Qu.:-375.0  
##  Median :-152.00   Median : 34.00   Median :601.0   Median :-319.0  
##  Mean   : -72.67   Mean   : 55.64   Mean   :593.8   Mean   :-345.3  
##  3rd Qu.:  27.00   3rd Qu.: 60.00   3rd Qu.:610.0   3rd Qu.:-306.0  
##  Max.   : 105.00   Max.   :479.00   Max.   :673.0   Max.   : 293.0  
##                                                                     
##     roll_arm         pitch_arm          yaw_arm         total_accel_arm
##  Min.   :-180.00   Min.   :-88.800   Min.   :-180.000   Min.   : 1.00  
##  1st Qu.: -31.00   1st Qu.:-26.000   1st Qu.: -42.900   1st Qu.:17.00  
##  Median :   0.00   Median :  0.000   Median :   0.000   Median :27.00  
##  Mean   :  18.04   Mean   : -4.599   Mean   :  -1.125   Mean   :25.56  
##  3rd Qu.:  77.20   3rd Qu.: 11.300   3rd Qu.:  45.300   3rd Qu.:33.00  
##  Max.   : 180.00   Max.   : 88.500   Max.   : 180.000   Max.   :66.00  
##                                                                        
##   gyros_arm_x        gyros_arm_y       gyros_arm_z       accel_arm_x     
##  Min.   :-6.37000   Min.   :-3.4000   Min.   :-2.3300   Min.   :-383.00  
##  1st Qu.:-1.32000   1st Qu.:-0.8000   1st Qu.:-0.0700   1st Qu.:-241.00  
##  Median : 0.08000   Median :-0.2600   Median : 0.2300   Median : -41.00  
##  Mean   : 0.04806   Mean   :-0.2587   Mean   : 0.2714   Mean   : -58.62  
##  3rd Qu.: 1.59000   3rd Qu.: 0.1400   3rd Qu.: 0.7200   3rd Qu.:  84.00  
##  Max.   : 4.87000   Max.   : 2.8400   Max.   : 3.0200   Max.   : 437.00  
##                                                                          
##   accel_arm_y       accel_arm_z       magnet_arm_x     magnet_arm_y   
##  Min.   :-318.00   Min.   :-636.00   Min.   :-584.0   Min.   :-392.0  
##  1st Qu.: -54.00   1st Qu.:-144.00   1st Qu.:-296.0   1st Qu.: -10.0  
##  Median :  14.00   Median : -46.00   Median : 294.0   Median : 200.0  
##  Mean   :  32.33   Mean   : -71.51   Mean   : 194.3   Mean   : 155.9  
##  3rd Qu.: 138.00   3rd Qu.:  23.00   3rd Qu.: 639.0   3rd Qu.: 322.0  
##  Max.   : 308.00   Max.   : 292.00   Max.   : 782.0   Max.   : 583.0  
##                                                                       
##   magnet_arm_z    roll_dumbbell     pitch_dumbbell     yaw_dumbbell     
##  Min.   :-597.0   Min.   :-153.71   Min.   :-148.50   Min.   :-150.871  
##  1st Qu.: 124.0   1st Qu.: -18.60   1st Qu.: -40.94   1st Qu.: -77.731  
##  Median : 443.0   Median :  48.35   Median : -21.01   Median :  -3.911  
##  Mean   : 304.5   Mean   :  23.67   Mean   : -10.90   Mean   :   1.520  
##  3rd Qu.: 545.0   3rd Qu.:  67.46   3rd Qu.:  17.22   3rd Qu.:  79.603  
##  Max.   : 694.0   Max.   : 153.55   Max.   : 149.40   Max.   : 154.952  
##                                                                         
##  total_accel_dumbbell gyros_dumbbell_x    gyros_dumbbell_y  
##  Min.   : 0.00        Min.   :-204.0000   Min.   :-2.10000  
##  1st Qu.: 5.00        1st Qu.:  -0.0300   1st Qu.:-0.14000  
##  Median :11.00        Median :   0.1400   Median : 0.03000  
##  Mean   :13.79        Mean   :   0.1579   Mean   : 0.04766  
##  3rd Qu.:20.00        3rd Qu.:   0.3500   3rd Qu.: 0.21000  
##  Max.   :58.00        Max.   :   2.2000   Max.   :52.00000  
##                                                             
##  gyros_dumbbell_z   accel_dumbbell_x accel_dumbbell_y  accel_dumbbell_z 
##  Min.   : -1.9500   Min.   :-419.0   Min.   :-179.00   Min.   :-334.00  
##  1st Qu.: -0.3100   1st Qu.: -50.0   1st Qu.:  -9.00   1st Qu.:-142.00  
##  Median : -0.1300   Median :  -9.0   Median :  43.00   Median :  -1.00  
##  Mean   : -0.1234   Mean   : -28.6   Mean   :  52.83   Mean   : -38.82  
##  3rd Qu.:  0.0300   3rd Qu.:  11.0   3rd Qu.: 111.00   3rd Qu.:  39.00  
##  Max.   :317.0000   Max.   : 235.0   Max.   : 315.00   Max.   : 318.00  
##                                                                         
##  magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z  roll_forearm    
##  Min.   :-643.0    Min.   :-744.0    Min.   :-250.00   Min.   :-180.00  
##  1st Qu.:-535.0    1st Qu.: 231.0    1st Qu.: -46.00   1st Qu.:  -0.67  
##  Median :-479.0    Median : 310.0    Median :  13.00   Median :  21.00  
##  Mean   :-325.5    Mean   : 219.1    Mean   :  45.65   Mean   :  33.98  
##  3rd Qu.:-295.0    3rd Qu.: 390.0    3rd Qu.:  94.00   3rd Qu.: 140.00  
##  Max.   : 584.0    Max.   : 632.0    Max.   : 452.00   Max.   : 180.00  
##                                                                         
##  pitch_forearm     yaw_forearm      total_accel_forearm gyros_forearm_x   
##  Min.   :-72.50   Min.   :-180.00   Min.   :  0.00      Min.   :-22.0000  
##  1st Qu.:  0.00   1st Qu.: -69.10   1st Qu.: 29.00      1st Qu.: -0.2200  
##  Median :  9.60   Median :   0.00   Median : 36.00      Median :  0.0500  
##  Mean   : 10.97   Mean   :  18.92   Mean   : 34.75      Mean   :  0.1596  
##  3rd Qu.: 28.70   3rd Qu.: 110.00   3rd Qu.: 41.00      3rd Qu.:  0.5800  
##  Max.   : 89.80   Max.   : 180.00   Max.   :108.00      Max.   :  3.9700  
##                                                                           
##  gyros_forearm_y     gyros_forearm_z    accel_forearm_x   accel_forearm_y 
##  Min.   : -7.02000   Min.   : -7.9400   Min.   :-498.00   Min.   :-632.0  
##  1st Qu.: -1.48000   1st Qu.: -0.1800   1st Qu.:-179.00   1st Qu.:  54.0  
##  Median :  0.03000   Median :  0.0800   Median : -57.00   Median : 201.0  
##  Mean   :  0.08152   Mean   :  0.1557   Mean   : -62.22   Mean   : 163.5  
##  3rd Qu.:  1.64000   3rd Qu.:  0.4900   3rd Qu.:  75.00   3rd Qu.: 312.0  
##  Max.   :311.00000   Max.   :231.0000   Max.   : 389.00   Max.   : 923.0  
##                                                                           
##  accel_forearm_z   magnet_forearm_x  magnet_forearm_y magnet_forearm_z
##  Min.   :-410.00   Min.   :-1280.0   Min.   :-896.0   Min.   :-966.0  
##  1st Qu.:-183.00   1st Qu.: -617.0   1st Qu.:  -3.0   1st Qu.: 194.0  
##  Median : -42.00   Median : -383.0   Median : 587.0   Median : 511.0  
##  Mean   : -56.63   Mean   : -314.7   Mean   : 375.8   Mean   : 395.6  
##  3rd Qu.:  25.00   3rd Qu.:  -77.0   3rd Qu.: 736.0   3rd Qu.: 652.0  
##  Max.   : 287.00   Max.   :  672.0   Max.   :1480.0   Max.   :1090.0  
##                                                                       
##  classe  
##  A:3906  
##  B:2658  
##  C:2396  
##  D:2252  
##  E:2525  
##          
## 
```

```r
sum(is.na(train_data$classe))
```

```
## [1] 0
```

```r
unique(train_data$classe)
```

```
## [1] A B C D E
## Levels: A B C D E
```
