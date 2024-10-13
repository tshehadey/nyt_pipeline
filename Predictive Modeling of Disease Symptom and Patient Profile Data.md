Final Project
================
Tysir Shehadey
2024-03-19

``` r
file_path <- "C:/Users/tysir/Disease_symptom_and_patient_profile_dataset.csv"
disease_df <- read.csv(file_path)

# Check the structure of the dataset
str(disease_df)
```

    ## 'data.frame':    349 obs. of  10 variables:
    ##  $ Disease             : chr  "Influenza" "Common Cold" "Eczema" "Asthma" ...
    ##  $ Fever               : chr  "Yes" "No" "No" "Yes" ...
    ##  $ Cough               : chr  "No" "Yes" "Yes" "Yes" ...
    ##  $ Fatigue             : chr  "Yes" "Yes" "Yes" "No" ...
    ##  $ Difficulty.Breathing: chr  "Yes" "No" "No" "Yes" ...
    ##  $ Age                 : int  19 25 25 25 25 25 25 25 28 28 ...
    ##  $ Gender              : chr  "Female" "Female" "Female" "Male" ...
    ##  $ Blood.Pressure      : chr  "Low" "Normal" "Normal" "Normal" ...
    ##  $ Cholesterol.Level   : chr  "Normal" "Normal" "Normal" "Normal" ...
    ##  $ Outcome.Variable    : chr  "Positive" "Negative" "Negative" "Positive" ...

``` r
# Print the column names
print(colnames(disease_df))
```

    ##  [1] "Disease"              "Fever"                "Cough"               
    ##  [4] "Fatigue"              "Difficulty.Breathing" "Age"                 
    ##  [7] "Gender"               "Blood.Pressure"       "Cholesterol.Level"   
    ## [10] "Outcome.Variable"

``` r
# Check for missing values in a data frame
missing_values <- is.na(disease_df)
summary(missing_values)
```

    ##   Disease          Fever           Cough          Fatigue       
    ##  Mode :logical   Mode :logical   Mode :logical   Mode :logical  
    ##  FALSE:349       FALSE:349       FALSE:349       FALSE:349      
    ##  Difficulty.Breathing    Age            Gender        Blood.Pressure 
    ##  Mode :logical        Mode :logical   Mode :logical   Mode :logical  
    ##  FALSE:349            FALSE:349       FALSE:349       FALSE:349      
    ##  Cholesterol.Level Outcome.Variable
    ##  Mode :logical     Mode :logical   
    ##  FALSE:349         FALSE:349

``` r
# Identify duplicate rows
duplicates <- duplicated(disease_df)

#Remove Duplicated Rows
disease_df <- disease_df[!duplicates,]
```

``` r
# EDA Visualizations

# Distribution of Age
ggplot(disease_df, aes(x = Age)) + 
  geom_histogram(binwidth = 5) + 
  ggtitle("Age Distribution") +
  xlab("Age") +
  ylab("Frequency")
```

![](Final-Project-ADS503_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
# Swarm Plot of Age by Outcome Variable
ggplot(disease_df, aes(x = Outcome.Variable, y = Age)) + 
  geom_jitter(width = 0.2) + 
  ggtitle("Swarm Plot of Age by Outcome Variable") + 
  xlab("Outcome Variable") + 
  ylab("Age")
```

![](Final-Project-ADS503_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
# Stacked Bar Chart of Gender and Fever by Outcome Variable
ggplot(disease_df, aes(x = Outcome.Variable, fill = interaction(Gender, Fever))) + 
  geom_bar(position = "fill") + 
  ggtitle("Stacked Bar Chart of Gender and Fever by Outcome Variable") +
  xlab("Outcome Variable") +
  ylab("Proportion") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_brewer(palette = "Set3", name = "Gender and Fever")
```

![](Final-Project-ADS503_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
# Define the disease categories
infectious_diseases <- c("Common Cold", "Influenza", "Chickenpox", "Measles", "HIV/AIDS", "Hepatitis B", "Tuberculosis", "Malaria", "Ebola Virus", "Zika Virus")
chronic_diseases <- c("Diabetes", "Hypertension", "Asthma", "Chronic Obstructive Pulmonary Disease (COPD)", "Chronic Kidney Disease", "Coronary Artery Disease")
genetic_disorders <- c("Down Syndrome", "Cystic Fibrosis", "Hemophilia", "Sickle Cell Anemia", "Klinefelter Syndrome", "Turner Syndrome", "Williams Syndrome")
cancers <- c("Breast Cancer", "Prostate Cancer", "Lung Cancer", "Colorectal Cancer", "Pancreatic Cancer", "Liver Cancer", "Kidney Cancer")
```

``` r
# Create a new column with the broader categories
disease_df <- disease_df %>%
  mutate(Category = case_when(
    Disease %in% infectious_diseases ~ "Infectious Diseases",
    Disease %in% chronic_diseases ~ "Chronic Diseases",
    Disease %in% genetic_disorders ~ "Genetic Disorders",
    Disease %in% cancers ~ "Cancers",
    TRUE ~ "Others"
  ))

# Print the first few rows to check the result
head(disease_df)
```

    ##       Disease Fever Cough Fatigue Difficulty.Breathing Age Gender
    ## 1   Influenza   Yes    No     Yes                  Yes  19 Female
    ## 2 Common Cold    No   Yes     Yes                   No  25 Female
    ## 3      Eczema    No   Yes     Yes                   No  25 Female
    ## 4      Asthma   Yes   Yes      No                  Yes  25   Male
    ## 6      Eczema   Yes    No      No                   No  25 Female
    ## 7   Influenza   Yes   Yes     Yes                  Yes  25 Female
    ##   Blood.Pressure Cholesterol.Level Outcome.Variable            Category
    ## 1            Low            Normal         Positive Infectious Diseases
    ## 2         Normal            Normal         Negative Infectious Diseases
    ## 3         Normal            Normal         Negative              Others
    ## 4         Normal            Normal         Positive    Chronic Diseases
    ## 6         Normal            Normal         Positive              Others
    ## 7         Normal            Normal         Positive Infectious Diseases

``` r
# Convert necessary columns to factors
disease_df$Fever <- as.factor(disease_df$Fever)
disease_df$Cough <- as.factor(disease_df$Cough)
disease_df$Fatigue <- as.factor(disease_df$Fatigue)
disease_df$Difficulty.Breathing <- as.factor(disease_df$Difficulty.Breathing)
disease_df$Gender <- as.factor(disease_df$Gender)
disease_df$Blood.Pressure <- as.factor(disease_df$Blood.Pressure)
disease_df$Cholesterol.Level <- as.factor(disease_df$Cholesterol.Level)
disease_df$Outcome.Variable <- as.factor(disease_df$Outcome.Variable)
disease_df$Category <- as.factor(disease_df$Category)

# Remove the original 'Disease' column as it is now encoded in 'Category'
disease_df <- disease_df %>% select(-Disease)
```

``` r
# Remove constant variables
constant_vars <- sapply(disease_df, function(x) length(unique(x)) == 1)
constant_vars <- names(disease_df)[constant_vars]
disease_df <- disease_df[, !names(disease_df) %in% constant_vars]
```

``` r
# Set seed for reproducibility
set.seed(123)
```

``` r
# Create a training and testing split with stratified sampling
trainIndex <- createDataPartition(disease_df$Category, p = 0.8, list = FALSE, times = 1)
dataTrain <- disease_df[trainIndex, ]
dataTest <- disease_df[-trainIndex, ]
```

``` r
# Ensure consistent levels
dataTrain$Category <- factor(dataTrain$Category)
dataTest$Category <- factor(dataTest$Category, levels = levels(dataTrain$Category))
```

``` r
#Assigning parameters to the Neural Network Model
nnetGrid <- expand.grid(size=1:5, decay=c(0, 0.1, 0.5, 1, 1.5, 1.8, 2))

#Train the Neural Network Model
set.seed(123)
nnetFit <- train(x = dataTrain[,1:9],
                 y = dataTrain$Category,
                 method = "nnet",
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 2000,
                 trControl = trainControl(method = "cv", number = 10))

#Neural Network Model
nnetFit
```

    ## Neural Network 
    ## 
    ## 243 samples
    ##   9 predictor
    ##   5 classes: 'Cancers', 'Chronic Diseases', 'Genetic Disorders', 'Infectious Diseases', 'Others' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 218, 217, 219, 218, 219, 220, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   size  decay  Accuracy   Kappa       
    ##   1     0.0    0.6876817   0.000000000
    ##   1     0.1    0.6876817   0.000000000
    ##   1     0.5    0.6876817   0.000000000
    ##   1     1.0    0.6876817   0.000000000
    ##   1     1.5    0.6876817   0.000000000
    ##   1     1.8    0.6876817   0.000000000
    ##   1     2.0    0.6876817   0.000000000
    ##   2     0.0    0.6833339   0.003379870
    ##   2     0.1    0.6716689  -0.011917755
    ##   2     0.5    0.6876817   0.000000000
    ##   2     1.0    0.6876817   0.000000000
    ##   2     1.5    0.6876817   0.000000000
    ##   2     1.8    0.6876817   0.000000000
    ##   2     2.0    0.6876817   0.000000000
    ##   3     0.0    0.6753484  -0.009281701
    ##   3     0.1    0.6586672  -0.020462981
    ##   3     0.5    0.6876817   0.000000000
    ##   3     1.0    0.6876817   0.000000000
    ##   3     1.5    0.6876817   0.000000000
    ##   3     1.8    0.6876817   0.000000000
    ##   3     2.0    0.6876817   0.000000000
    ##   4     0.0    0.6501254   0.004335218
    ##   4     0.1    0.6628066   0.023337780
    ##   4     0.5    0.6876817   0.000000000
    ##   4     1.0    0.6876817   0.000000000
    ##   4     1.5    0.6876817   0.000000000
    ##   4     1.8    0.6876817   0.000000000
    ##   4     2.0    0.6876817   0.000000000
    ##   5     0.0    0.6501382   0.031969339
    ##   5     0.1    0.6470022   0.023289858
    ##   5     0.5    0.6876817   0.000000000
    ##   5     1.0    0.6876817   0.000000000
    ##   5     1.5    0.6876817   0.000000000
    ##   5     1.8    0.6876817   0.000000000
    ##   5     2.0    0.6876817   0.000000000
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were size = 1 and decay = 2.

``` r
nnetFit$finalModel
```

    ## a 11-1-5 network with 22 weights
    ## inputs: FeverYes CoughYes FatigueYes Difficulty.BreathingYes Age GenderMale Blood.PressureLow Blood.PressureNormal Cholesterol.LevelLow Cholesterol.LevelNormal Outcome.VariablePositive 
    ## output(s): .outcome 
    ## options were - softmax modelling  decay=2

``` r
#Predict on Test set
nnet_predictions <- predict(nnetFit, newdata = dataTest[,1:9])

#Confusion Matrix for Neural Network
nnetCM <- confusionMatrix(nnet_predictions, dataTest$Category)

nnetCM
```

    ## Confusion Matrix and Statistics
    ## 
    ##                      Reference
    ## Prediction            Cancers Chronic Diseases Genetic Disorders
    ##   Cancers                   0                0                 0
    ##   Chronic Diseases          0                0                 0
    ##   Genetic Disorders         0                0                 0
    ##   Infectious Diseases       0                0                 0
    ##   Others                    3                8                 1
    ##                      Reference
    ## Prediction            Infectious Diseases Others
    ##   Cancers                               0      0
    ##   Chronic Diseases                      0      0
    ##   Genetic Disorders                     0      0
    ##   Infectious Diseases                   0      0
    ##   Others                                4     41
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7193          
    ##                  95% CI : (0.5846, 0.8303)
    ##     No Information Rate : 0.7193          
    ##     P-Value [Acc > NIR] : 0.5669          
    ##                                           
    ##                   Kappa : 0               
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: Cancers Class: Chronic Diseases
    ## Sensitivity                 0.00000                  0.0000
    ## Specificity                 1.00000                  1.0000
    ## Pos Pred Value                  NaN                     NaN
    ## Neg Pred Value              0.94737                  0.8596
    ## Prevalence                  0.05263                  0.1404
    ## Detection Rate              0.00000                  0.0000
    ## Detection Prevalence        0.00000                  0.0000
    ## Balanced Accuracy           0.50000                  0.5000
    ##                      Class: Genetic Disorders Class: Infectious Diseases
    ## Sensitivity                           0.00000                    0.00000
    ## Specificity                           1.00000                    1.00000
    ## Pos Pred Value                            NaN                        NaN
    ## Neg Pred Value                        0.98246                    0.92982
    ## Prevalence                            0.01754                    0.07018
    ## Detection Rate                        0.00000                    0.00000
    ## Detection Prevalence                  0.00000                    0.00000
    ## Balanced Accuracy                     0.50000                    0.50000
    ##                      Class: Others
    ## Sensitivity                 1.0000
    ## Specificity                 0.0000
    ## Pos Pred Value              0.7193
    ## Neg Pred Value                 NaN
    ## Prevalence                  0.7193
    ## Detection Rate              0.7193
    ## Detection Prevalence        1.0000
    ## Balanced Accuracy           0.5000

``` r
# Ensure 'Category' is a factor
dataTrain$Category <- as.factor(dataTrain$Category)
dataTest$Category <- as.factor(dataTest$Category)

# Create dummy variables for categorical predictors (Train)
dummies <- dummyVars(~ ., data = dataTrain[, !names(dataTrain) %in% "Category"])
dataTrain_transformed <- predict(dummies, newdata = dataTrain)

# Convert the transformed data to a data frame and add the Category column back (Train)
dataTrain_transformed <- data.frame(dataTrain_transformed)
dataTrain_transformed$Category <- dataTrain$Category

# Create dummy variables for categorical predictors (Test)
dummies2 <- dummyVars(~ ., data = dataTest[, !names(dataTest) %in% "Category"])
dataTest_transformed <- predict(dummies2, newdata = dataTest)

# Convert the transformed data to a data frame and add the Category column back (Test)
dataTest_transformed <- data.frame(dataTest_transformed)
dataTest_transformed$Category <- dataTest$Category

#Train the KNN Model
set.seed(123)
knnFit <- train(x = dataTrain_transformed[,1:9],
                y = dataTrain_transformed$Category,
                method = "knn",
                tuneLength =10,
                trControl = trainControl(method = "cv", number = 10))

knnFit
```

    ## k-Nearest Neighbors 
    ## 
    ## 243 samples
    ##   9 predictor
    ##   5 classes: 'Cancers', 'Chronic Diseases', 'Genetic Disorders', 'Infectious Diseases', 'Others' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 218, 217, 219, 218, 219, 220, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   k   Accuracy   Kappa        
    ##    5  0.6709732   0.0009536137
    ##    7  0.6631544  -0.0329847427
    ##    9  0.6791672   0.0002644591
    ##   11  0.6876817   0.0000000000
    ##   13  0.6876817   0.0000000000
    ##   15  0.6876817   0.0000000000
    ##   17  0.6876817   0.0000000000
    ##   19  0.6876817   0.0000000000
    ##   21  0.6876817   0.0000000000
    ##   23  0.6876817   0.0000000000
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was k = 23.

``` r
#Predict on Test set
knn_predictions <- predict(knnFit, newdata = dataTest_transformed[,1:9])

#Confusion Matrix for Neural Network
knnCM <- confusionMatrix(knn_predictions, dataTest_transformed$Category)

knnCM
```

    ## Confusion Matrix and Statistics
    ## 
    ##                      Reference
    ## Prediction            Cancers Chronic Diseases Genetic Disorders
    ##   Cancers                   0                0                 0
    ##   Chronic Diseases          0                0                 0
    ##   Genetic Disorders         0                0                 0
    ##   Infectious Diseases       0                0                 0
    ##   Others                    3                8                 1
    ##                      Reference
    ## Prediction            Infectious Diseases Others
    ##   Cancers                               0      0
    ##   Chronic Diseases                      0      0
    ##   Genetic Disorders                     0      0
    ##   Infectious Diseases                   0      0
    ##   Others                                4     41
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7193          
    ##                  95% CI : (0.5846, 0.8303)
    ##     No Information Rate : 0.7193          
    ##     P-Value [Acc > NIR] : 0.5669          
    ##                                           
    ##                   Kappa : 0               
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: Cancers Class: Chronic Diseases
    ## Sensitivity                 0.00000                  0.0000
    ## Specificity                 1.00000                  1.0000
    ## Pos Pred Value                  NaN                     NaN
    ## Neg Pred Value              0.94737                  0.8596
    ## Prevalence                  0.05263                  0.1404
    ## Detection Rate              0.00000                  0.0000
    ## Detection Prevalence        0.00000                  0.0000
    ## Balanced Accuracy           0.50000                  0.5000
    ##                      Class: Genetic Disorders Class: Infectious Diseases
    ## Sensitivity                           0.00000                    0.00000
    ## Specificity                           1.00000                    1.00000
    ## Pos Pred Value                            NaN                        NaN
    ## Neg Pred Value                        0.98246                    0.92982
    ## Prevalence                            0.01754                    0.07018
    ## Detection Rate                        0.00000                    0.00000
    ## Detection Prevalence                  0.00000                    0.00000
    ## Balanced Accuracy                     0.50000                    0.50000
    ##                      Class: Others
    ## Sensitivity                 1.0000
    ## Specificity                 0.0000
    ## Pos Pred Value              0.7193
    ## Neg Pred Value                 NaN
    ## Prevalence                  0.7193
    ## Detection Rate              0.7193
    ## Detection Prevalence        1.0000
    ## Balanced Accuracy           0.5000

``` r
# Define trainControl
train_control <- trainControl(method = "cv", number = 10)
```

``` r
# Train a Gradient Boosting Machine model
set.seed(123)
gbm_model <- train(Category ~ ., data = dataTrain, method = "gbm", verbose = FALSE, trControl = trainControl(method = "cv", number = 10))
```

``` r
# Predict on test set
gbm_predictions <- predict(gbm_model, newdata = dataTest)
```

``` r
confusion_matrix <- confusionMatrix(gbm_predictions, dataTest$Category)
print(confusion_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##                      Reference
    ## Prediction            Cancers Chronic Diseases Genetic Disorders
    ##   Cancers                   0                0                 0
    ##   Chronic Diseases          0                0                 0
    ##   Genetic Disorders         0                0                 0
    ##   Infectious Diseases       1                0                 0
    ##   Others                    2                8                 1
    ##                      Reference
    ## Prediction            Infectious Diseases Others
    ##   Cancers                               0      0
    ##   Chronic Diseases                      0      6
    ##   Genetic Disorders                     0      0
    ##   Infectious Diseases                   0      0
    ##   Others                                4     35
    ## 
    ## Overall Statistics
    ##                                         
    ##                Accuracy : 0.614         
    ##                  95% CI : (0.4757, 0.74)
    ##     No Information Rate : 0.7193        
    ##     P-Value [Acc > NIR] : 0.9693        
    ##                                         
    ##                   Kappa : -0.0933       
    ##                                         
    ##  Mcnemar's Test P-Value : NA            
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: Cancers Class: Chronic Diseases
    ## Sensitivity                 0.00000                  0.0000
    ## Specificity                 1.00000                  0.8776
    ## Pos Pred Value                  NaN                  0.0000
    ## Neg Pred Value              0.94737                  0.8431
    ## Prevalence                  0.05263                  0.1404
    ## Detection Rate              0.00000                  0.0000
    ## Detection Prevalence        0.00000                  0.1053
    ## Balanced Accuracy           0.50000                  0.4388
    ##                      Class: Genetic Disorders Class: Infectious Diseases
    ## Sensitivity                           0.00000                    0.00000
    ## Specificity                           1.00000                    0.98113
    ## Pos Pred Value                            NaN                    0.00000
    ## Neg Pred Value                        0.98246                    0.92857
    ## Prevalence                            0.01754                    0.07018
    ## Detection Rate                        0.00000                    0.00000
    ## Detection Prevalence                  0.00000                    0.01754
    ## Balanced Accuracy                     0.50000                    0.49057
    ##                      Class: Others
    ## Sensitivity                 0.8537
    ## Specificity                 0.0625
    ## Pos Pred Value              0.7000
    ## Neg Pred Value              0.1429
    ## Prevalence                  0.7193
    ## Detection Rate              0.6140
    ## Detection Prevalence        0.8772
    ## Balanced Accuracy           0.4581

``` r
# Train a Support Vector Machine model
set.seed(123)
svm_model <- train(Category ~ ., data = dataTrain, method = "svmRadial", trControl = trainControl(method = "cv", number = 10))
```

``` r
# Predict on test set using SVM model
svm_predictions <- predict(svm_model, newdata = dataTest)
```

``` r
# Evaluate the SVM model
svm_confusion_matrix <- confusionMatrix(svm_predictions, dataTest$Category)
print(svm_confusion_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##                      Reference
    ## Prediction            Cancers Chronic Diseases Genetic Disorders
    ##   Cancers                   0                0                 0
    ##   Chronic Diseases          0                0                 0
    ##   Genetic Disorders         0                0                 0
    ##   Infectious Diseases       0                0                 0
    ##   Others                    3                8                 1
    ##                      Reference
    ## Prediction            Infectious Diseases Others
    ##   Cancers                               0      0
    ##   Chronic Diseases                      0      0
    ##   Genetic Disorders                     0      0
    ##   Infectious Diseases                   0      0
    ##   Others                                4     41
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7193          
    ##                  95% CI : (0.5846, 0.8303)
    ##     No Information Rate : 0.7193          
    ##     P-Value [Acc > NIR] : 0.5669          
    ##                                           
    ##                   Kappa : 0               
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: Cancers Class: Chronic Diseases
    ## Sensitivity                 0.00000                  0.0000
    ## Specificity                 1.00000                  1.0000
    ## Pos Pred Value                  NaN                     NaN
    ## Neg Pred Value              0.94737                  0.8596
    ## Prevalence                  0.05263                  0.1404
    ## Detection Rate              0.00000                  0.0000
    ## Detection Prevalence        0.00000                  0.0000
    ## Balanced Accuracy           0.50000                  0.5000
    ##                      Class: Genetic Disorders Class: Infectious Diseases
    ## Sensitivity                           0.00000                    0.00000
    ## Specificity                           1.00000                    1.00000
    ## Pos Pred Value                            NaN                        NaN
    ## Neg Pred Value                        0.98246                    0.92982
    ## Prevalence                            0.01754                    0.07018
    ## Detection Rate                        0.00000                    0.00000
    ## Detection Prevalence                  0.00000                    0.00000
    ## Balanced Accuracy                     0.50000                    0.50000
    ##                      Class: Others
    ## Sensitivity                 1.0000
    ## Specificity                 0.0000
    ## Pos Pred Value              0.7193
    ## Neg Pred Value                 NaN
    ## Prevalence                  0.7193
    ## Detection Rate              0.7193
    ## Detection Prevalence        1.0000
    ## Balanced Accuracy           0.5000

``` r
# Train a Random Forest model
set.seed(123)
rf_model <- train(Category ~ ., data = dataTrain, method = "rf", trControl = trainControl(method = "cv", number = 10))

# Print the Random Forest model
print(rf_model)
```

    ## Random Forest 
    ## 
    ## 243 samples
    ##   9 predictor
    ##   5 classes: 'Cancers', 'Chronic Diseases', 'Genetic Disorders', 'Infectious Diseases', 'Others' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 218, 217, 219, 218, 219, 220, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa     
    ##    2    0.6876817  0.00000000
    ##    6    0.6428612  0.07178324
    ##   11    0.6180134  0.06217868
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 2.

``` r
# Predict on test set using Random Forest model
rf_predictions <- predict(rf_model, newdata = dataTest)

# Evaluate the Random Forest model
rf_confusion_matrix <- confusionMatrix(rf_predictions, dataTest$Category)
print(rf_confusion_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##                      Reference
    ## Prediction            Cancers Chronic Diseases Genetic Disorders
    ##   Cancers                   0                0                 0
    ##   Chronic Diseases          0                0                 0
    ##   Genetic Disorders         0                0                 0
    ##   Infectious Diseases       0                0                 0
    ##   Others                    3                8                 1
    ##                      Reference
    ## Prediction            Infectious Diseases Others
    ##   Cancers                               0      0
    ##   Chronic Diseases                      0      0
    ##   Genetic Disorders                     0      0
    ##   Infectious Diseases                   0      0
    ##   Others                                4     41
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7193          
    ##                  95% CI : (0.5846, 0.8303)
    ##     No Information Rate : 0.7193          
    ##     P-Value [Acc > NIR] : 0.5669          
    ##                                           
    ##                   Kappa : 0               
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: Cancers Class: Chronic Diseases
    ## Sensitivity                 0.00000                  0.0000
    ## Specificity                 1.00000                  1.0000
    ## Pos Pred Value                  NaN                     NaN
    ## Neg Pred Value              0.94737                  0.8596
    ## Prevalence                  0.05263                  0.1404
    ## Detection Rate              0.00000                  0.0000
    ## Detection Prevalence        0.00000                  0.0000
    ## Balanced Accuracy           0.50000                  0.5000
    ##                      Class: Genetic Disorders Class: Infectious Diseases
    ## Sensitivity                           0.00000                    0.00000
    ## Specificity                           1.00000                    1.00000
    ## Pos Pred Value                            NaN                        NaN
    ## Neg Pred Value                        0.98246                    0.92982
    ## Prevalence                            0.01754                    0.07018
    ## Detection Rate                        0.00000                    0.00000
    ## Detection Prevalence                  0.00000                    0.00000
    ## Balanced Accuracy                     0.50000                    0.50000
    ##                      Class: Others
    ## Sensitivity                 1.0000
    ## Specificity                 0.0000
    ## Pos Pred Value              0.7193
    ## Neg Pred Value                 NaN
    ## Prevalence                  0.7193
    ## Detection Rate              0.7193
    ## Detection Prevalence        1.0000
    ## Balanced Accuracy           0.5000

``` r
# Train a Penalized Logistic Regression model (Lasso)
set.seed(123)
pen_log_reg_model <- train(Category ~ ., data = dataTrain, method = "glmnet", trControl = trainControl(method = "cv", number = 10))
```

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

    ## Warning in lognet(xd, is.sparse, ix, jx, y, weights, offset, alpha, nobs, : one
    ## multinomial or binomial class has fewer than 8 observations; dangerous ground

``` r
# Print the Penalized Logistic Regression model
print(pen_log_reg_model)
```

    ## glmnet 
    ## 
    ## 243 samples
    ##   9 predictor
    ##   5 classes: 'Cancers', 'Chronic Diseases', 'Genetic Disorders', 'Infectious Diseases', 'Others' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 218, 217, 219, 218, 219, 220, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   alpha  lambda       Accuracy   Kappa       
    ##   0.10   0.000126755  0.6878484   0.034136716
    ##   0.10   0.001267550  0.6878484   0.034136716
    ##   0.10   0.012675501  0.6836817  -0.004651163
    ##   0.55   0.000126755  0.6878484   0.034136716
    ##   0.55   0.001267550  0.6836817   0.010500352
    ##   0.55   0.012675501  0.6836817  -0.004651163
    ##   1.00   0.000126755  0.6878484   0.034136716
    ##   1.00   0.001267550  0.6836817   0.010500352
    ##   1.00   0.012675501  0.6876817   0.000000000
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were alpha = 0.1 and lambda = 0.00126755.

``` r
# Predict on test set using Penalized Logistic Regression model
pen_log_reg_predictions <- predict(pen_log_reg_model, newdata = dataTest)
# Evaluate the Penalized Logistic Regression model
pen_log_reg_confusion_matrix <- confusionMatrix(pen_log_reg_predictions, dataTest$Category)
print(pen_log_reg_confusion_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##                      Reference
    ## Prediction            Cancers Chronic Diseases Genetic Disorders
    ##   Cancers                   0                0                 0
    ##   Chronic Diseases          0                2                 0
    ##   Genetic Disorders         0                0                 0
    ##   Infectious Diseases       0                0                 0
    ##   Others                    3                6                 1
    ##                      Reference
    ## Prediction            Infectious Diseases Others
    ##   Cancers                               0      0
    ##   Chronic Diseases                      0      1
    ##   Genetic Disorders                     0      0
    ##   Infectious Diseases                   0      0
    ##   Others                                4     40
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7368          
    ##                  95% CI : (0.6034, 0.8446)
    ##     No Information Rate : 0.7193          
    ##     P-Value [Acc > NIR] : 0.4499          
    ##                                           
    ##                   Kappa : 0.1543          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: Cancers Class: Chronic Diseases
    ## Sensitivity                 0.00000                 0.25000
    ## Specificity                 1.00000                 0.97959
    ## Pos Pred Value                  NaN                 0.66667
    ## Neg Pred Value              0.94737                 0.88889
    ## Prevalence                  0.05263                 0.14035
    ## Detection Rate              0.00000                 0.03509
    ## Detection Prevalence        0.00000                 0.05263
    ## Balanced Accuracy           0.50000                 0.61480
    ##                      Class: Genetic Disorders Class: Infectious Diseases
    ## Sensitivity                           0.00000                    0.00000
    ## Specificity                           1.00000                    1.00000
    ## Pos Pred Value                            NaN                        NaN
    ## Neg Pred Value                        0.98246                    0.92982
    ## Prevalence                            0.01754                    0.07018
    ## Detection Rate                        0.00000                    0.00000
    ## Detection Prevalence                  0.00000                    0.00000
    ## Balanced Accuracy                     0.50000                    0.50000
    ##                      Class: Others
    ## Sensitivity                 0.9756
    ## Specificity                 0.1250
    ## Pos Pred Value              0.7407
    ## Neg Pred Value              0.6667
    ## Prevalence                  0.7193
    ## Detection Rate              0.7018
    ## Detection Prevalence        0.9474
    ## Balanced Accuracy           0.5503

``` r
# Neural Network Accuracy
nnet_accuracy <- nnetCM$overall['Accuracy']
print(paste("Neural Network Accuracy: ", nnet_accuracy))
```

    ## [1] "Neural Network Accuracy:  0.719298245614035"

``` r
# KNN Accuracy
knn_accuracy <- knnCM$overall['Accuracy']
print(paste("KNN Accuracy: ", knn_accuracy))
```

    ## [1] "KNN Accuracy:  0.719298245614035"

``` r
# Gradient Boosting Machine Accuracy
gbm_accuracy <- confusion_matrix$overall['Accuracy']
print(paste("Gradient Boosting Machine Accuracy: ", gbm_accuracy))
```

    ## [1] "Gradient Boosting Machine Accuracy:  0.614035087719298"

``` r
# Support Vector Machine Accuracy
svm_accuracy <- svm_confusion_matrix$overall['Accuracy']
print(paste("Support Vector Machine Accuracy: ", svm_accuracy))
```

    ## [1] "Support Vector Machine Accuracy:  0.719298245614035"

``` r
# Random Forest Accuracy
rf_accuracy <- rf_confusion_matrix$overall['Accuracy']
print(paste("Random Forest Accuracy: ", rf_accuracy))
```

    ## [1] "Random Forest Accuracy:  0.719298245614035"

``` r
# Penalized Logistic Regression Accuracy
pen_log_reg_accuracy <- pen_log_reg_confusion_matrix$overall['Accuracy']
print(paste("Penalized Logistic Regression Accuracy: ", pen_log_reg_accuracy))
```

    ## [1] "Penalized Logistic Regression Accuracy:  0.736842105263158"

``` r
# Store accuracies in a named vector
accuracy_scores <- c(
  "Neural Network" = nnet_accuracy,
  "KNN" = knn_accuracy,
  "Gradient Boosting Machine" = gbm_accuracy,
  "Support Vector Machine" = svm_accuracy,
  "Random Forest" = rf_accuracy,
  "Penalized Logistic Regression" = pen_log_reg_accuracy
)

# Find the best model based on accuracy
best_model <- names(accuracy_scores)[which.max(accuracy_scores)]
print(paste("The best model is:", best_model, " with an accuracy of", max(accuracy_scores)))
```

    ## [1] "The best model is: Penalized Logistic Regression.Accuracy  with an accuracy of 0.736842105263158"
