                                                Authors: Venkat Koneru and Tharun Reddy
                                                
# Weather Prediction

This project leverages machine learning algorithms to forecast weather conditions. By analyzing historical weather data we predict the future trends in the weather. Our goal is to provide reliable and actionable weather forecasts to help individuals and organizations make informed decisions. 

## Process Overview

#### Data Inspection and Cleaning: 
Checked for missing values, duplicates, and inspected data types and unique values.

#### Reordering Columns: 
Reordered the columns for better organization and readability.

#### Data Visualization:
Plotted histograms and box plots for numeric features to understand their distributions.
Created count plots for categorical features to visualize their frequency distributions.
Target Analysis: Analyzed the features with respect to the target variable using count plots and box plots.

#### One-Hot Encoding and Factorizing: 
Applied one-hot encoding to categorical features and factorized the target variable for numerical representation.

#### Correlation Analysis: 
Generated a correlation matrix to understand the relationships between features.

#### Data Splitting: 
Split the dataset into training and testing sets to evaluate model performance.

#### Feature Scaling: 
Applied MinMaxScaler to normalize the feature values for better model performance.

#### Model Implementation:
Trained multiple models including Logistic Regression, Decision Tree Classifier, Random Forest Classifier, and Support Vector Machine (SVM).
Predicted the target variable using the trained models and evaluated their accuracy. 

#### Hyperparameter Tuning: 
Used GridSearchCV to perform hyperparameter tuning for each model to find the best parameters and improve model performance.
 
Following this structured process, we aim to develop models for weather prediction. Alongside, leveraging hyperparameter tuning and comparative analysis to achieve optimal performance and reliability.


# EDA
[Kaggle](https://www.kaggle.com/datasets/nikhil7280/weather-type-classification/data)

#### Dataset Description
This dataset is synthetically generated to mimic weather data for classification tasks. It has 13200 samples and 11 columns
##### Attributes
Temperature (numeric): The temperature in degrees Celsius, ranging from extreme cold to extreme heat.
Humidity (numeric): The humidity percentage, including values above 100% to introduce outliers.
Wind Speed (numeric): The wind speed in kilometers per hour, with a range including unrealistically high values.
Precipitation (%) (numeric): The precipitation percentage, including outlier values.
Cloud Cover (categorical): The cloud cover description.
Atmospheric Pressure (numeric): The atmospheric pressure in hPa, covering a wide range.
UV Index (numeric): The UV index, indicating the strength of ultraviolet radiation.
Season (categorical): The season during which the data was recorded.
Visibility (km) (numeric): The visibility in kilometers, including very low or very high values.
Location (categorical): The type of location where the data was recorded.

![Temperature distribution](https://github.com/user-attachments/assets/47fe4f00-a262-4242-86a3-093991c44fee)


![Cloud Cover distribution](https://github.com/user-attachments/assets/10535629-de1e-4184-a26f-aee847d4044f)

Output/y: Weather Type (categorical) -> The target variable for classification, indicating the weather type.

![Seasons Distribution](https://github.com/user-attachments/assets/ee15adaa-20d5-440f-9804-e03e2d091662)
    
Problem type: Classification     

## Correlation among attributes

![Correlation Matrix](https://github.com/user-attachments/assets/9f69efc5-c55a-45dc-bb5c-04b324b2c0f1)

#### Train/Test Splitting:
- We chose to use standard convention of 70:30 spplit for splitting the dataset into training and testing data.

#### Model Selection: 
We have used Logistic Regression, Random Forect, Support Vector Machine and Decision Tree to compare and contrast our observations.

## Validation / metrics

- Accuracy provides a comprehensive measure of a model's overall performance across all classes. It gives a single numerical value that summarizes the correctness of predictions.

#### Logistic Regression :
![confusion matrix](https://github.com/user-attachments/assets/33de5a4c-1390-4cc5-ae33-57c5f0780023)
![Roc curve](https://github.com/user-attachments/assets/1715871a-26d3-4822-b3da-808bbd60f4c8)

#### SVM:
![Confusion Matrix](https://github.com/user-attachments/assets/40128a98-fc4d-4d47-847d-aa104b9b9d7e)
![ROC Curve](https://github.com/user-attachments/assets/08b562cb-1c19-4ccb-af64-af74aab0c8e6)

#### Random Forest
![Confusion Matrix](https://github.com/user-attachments/assets/fb702d9e-6ef2-4af5-a505-4e573174323c)
![ROC Curve](https://github.com/user-attachments/assets/16717c49-90c3-4dfc-8578-18d089eb22ae)

#### Decision Tree
![Confusion Matrix](https://github.com/user-attachments/assets/b8549f11-8e11-43bd-b858-6ceb0ae1edf1)
![ROC Curve](https://github.com/user-attachments/assets/c2ba0c8e-4137-47de-b80e-52bb4c1ab986)

## Observations:
Logistic Regression Accuracy: 86.97%
Random Forest Accuracy: 
Decision Tree Accuracy:
SVM Accuracy: 

## Production
This can be used for commerical purposes such as agriculture, defence, rocket launch, transportation.  

