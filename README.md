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

    
Output/y: Weather Type (categorical) -> The target variable for classification, indicating the weather type.

Problem type: Classification     


## Model fitting

#### Train/Test Splitting:
- We chose to use standard convention of 70:30 spplit for splitting the dataset into training and testing data.

#### Model Selection: 



## Validation / metrics

- Accuracy provides a comprehensive measure of a model's overall performance across all classes. It gives a single numerical value that summarizes the correctness of predictions.

Logistic Regression:
![Confusion matrix](https://github.com/nipun-davasam/IA651-Applied-Machine-Learning/assets/151178533/882e698e-ccbc-4ab9-8ba1-f716b8614ac4)

Misclassified Samples:
![False pred](https://github.com/nipun-davasam/IA651-Applied-Machine-Learning/assets/151178533/91faa030-861c-482c-998e-e75aa88d232d)


![Confusion matrix](https://github.com/nipun-davasam/IA651-Applied-Machine-Learning/assets/151178533/ab34d377-e7b1-4789-b9d9-65c11f72cb57)

- Our model correctly predicts positives with a true positive rate of 98.3% and a false positive rate of 2.8%.

## Observations:



## Production

Mobile applications for remote diagnosis: Mobile apps equipped with malaria detection models can enable individuals in remote or rural areas to perform self-testing for malaria using their smartphones. The app can capture images of blood smears or use rapid diagnostic tests (RDTs) to provide preliminary diagnoses, which can then be verified by healthcare professionals.

