# %%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px

# %%
from sklearn.preprocessing import MinMaxScaler,label_binarize
from sklearn.model_selection import train_test_split,GridSearchCV,validation_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score,auc,roc_curve,classification_report

# %%
data = pd.read_csv("weather_classification_data.csv")
data

# %%
print(data.isna().sum())
print(data.duplicated().sum())
print(data.info())


print(data["Weather Type"].unique())




# %%
print(data.describe()) 


# %%
# data.columns
new_order  = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index','Visibility (km)','Cloud Cover', 'Season', 'Location', 'Weather Type']
data = data[new_order]

data["Temperature"].min()



# %% [markdown]
# ## EDA

# %% [markdown]
# The following plot illustrates the frequency distribution and summary of data, focusing exclusively on numerical variables.
# 
# Histogram: This plot shows how data points are distributed across different values, providing a visual representation of the frequency of data points within specified ranges.
# Box Plot: This plot highlights the interquartile range (IQR), indicating how data is concentrated within this range, and also displays the median, offering insights into the central tendency of the dataset
# 
# In our dataset, the integer-type variables are Temperature, Humidity, Wind Speed, and Precipitation. We observed that Wind Speed and Precipitation contain outliers. These outliers represent rare cases, and it is uncertain whether such extreme values for Temperature and Wind Speed will recur. Therefore, we chose not to handle or remove these outliers.

# %%
def Plot_1(data, col):
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(data[col], bins=10, edgecolor='black',color= 'skyblue')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title(f'{col} Distribution')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(data[col])
    plt.ylabel(col)
    plt.title(f'{col} BoxPlot')
    
    plt.tight_layout()
    plt.show()

for i in range(4):
    Plot_1(data, data.columns[i])


# %% [markdown]
# The following plot is a bar plot analysis of categorical variables within the dataset, clearly showing the percentage of data points for each category. The dataset includes three categorical columns: Cloud Cover, Season, and Location. Although Weather Type is also present, it serves as our target variable.
# 
# This count plot visually represents the frequency of different categories for Cloud Cover, Season, and Location in the dataset. The bars are sorted by percentage, allowing us to easily verify which types of cloud cover, locations, and seasons are most prevalent in the data.

# %%
def Plot_2(data, col):

    palette = sns.color_palette("Set1", len(data[col].unique()))
    count = sns.countplot(data=data, x=col, order=data[col].value_counts().index,palette= palette)
    total = len(data)
    
    for p in count.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        count.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')

    plt.show()

for i in range(-4,-1):
    Plot_2(data,data.columns[i])

# %% [markdown]
# the below plot shows the visulization about my X varaible with my y variable so hear my y = weather type i have just plot the colored compared plot with below with all other variable 

# %%
target = data["Weather Type"]

def Plot_3(data, col):
    if data[col].dtype == 'object':
        plt.figure(figsize=(10, 6))
        sns.countplot(data=data, x=col, hue=target, palette='hot', order=data[col].value_counts().index)
        plt.title(f'Count Plot of {col}')
        plt.show()
    else:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=data, x=target, y=col, palette='hot')
        plt.title(f'Box Plot of {col}')
        plt.show()

[Plot_3(data,col) for col in data.columns]


# %% [markdown]
# Hear we perform one hot encoding for catagorical variable i.e(Cloud cover,Season,location)and for taget varaible we perform label encoding because the above above weather type plot explain me the data of 

# %% [markdown]
# ## One hot encoding

# %% [markdown]
# For the columns ‘Cloud Cover’, ‘Season’, and ‘Location’, I performed one-hot encoding to handle the categorical data. The categories within these columns have different relationships with each other, but they all influence the weather type. The data appears balanced across these categories, as indicated by the above plot. This ensures that the model can effectively learn from all relevant features without bias towards any particular category.
# 
# Additionally, one-hot encoding helps in converting categorical variables into a format that can be provided to ML algorithms to improve predictions. By including all these encoded features, we ensure that the model captures the nuanced relationships between different weather conditions and the target variable.

# %%
data_1 = pd.get_dummies(data, columns=['Cloud Cover', 'Season', 'Location']) # one hot encoding 

data_1["Weather Type"] = pd.factorize(data["Weather Type"])[0] +1 # factorize 

data_1


# %% [markdown]
# The correlation plot below illustrates the relationships between each column in the dataset. It clearly shows that humidity, wind speed, and precipitation are highly positively correlated. However, I have considered all elements in the dataset for analysis, as each variable has some relationship with the weather type. Therefore, I included all variables in the model operation.

# %%
correlation= data_1.corr()
fig = px.imshow(correlation,
                text_auto = True,
                aspect = "auto",
                color_continuous_scale='YlOrRd',
                title='Correlation Matrix')

fig.update_layout(xaxis = dict(tickangle = -45),width = 1500,height = 900)
fig.show()

# %% [markdown]
# ### Train test split operation 

# %%
X = data_1.drop("Weather Type",axis = 1)
y = data_1['Weather Type']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state= 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# %% [markdown]
# ### Scaling the splitted dataset 
# 

# %%
Scalar = MinMaxScaler()
X_train = Scalar.fit_transform(X_train)
X_test = Scalar.transform(X_test)
X_test

# %%
def Plot_5(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(
        conf_matrix,
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        x=['Rainy', 'Cloudy', 'Sunny', 'Snowy'],
        y=['Rainy', 'Cloudy', 'Sunny', 'Snowy'],
        text_auto=True,
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(title='Confusion Matrix',width = 800,height = 400)
    fig.show()


# %% [markdown]
# ### LogisticRegression

# %%

Model_LR = LogisticRegression(solver='saga')
Model_LR.fit(X_train, y_train)

LR_Paramater_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

LR_Paramater_Search = GridSearchCV(estimator=LogisticRegression(solver = 'saga'),
                                   param_grid= LR_Paramater_grid,
                                   cv = 5,scoring="accuracy")
LR_Paramater_Search.fit(X_train,y_train)
y_pred_LR = LR_Paramater_Search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_LR)
Confusion_LR = confusion_matrix(y_test, y_pred_LR)


# %%
print("Logistic Regression - Best Params:", LR_Paramater_Search.best_params_)
print("Logistic Regression - Best Estimator:",LR_Paramater_Search.best_estimator_)
print("Logistic Regression - Best Score:",LR_Paramater_Search.best_score_)
print(Confusion_LR)
print("Accuracy:", accuracy)
Plot_5(y_test, y_pred_LR)

# %% [markdown]
# ### Decision Tree

# %%
Model_DT = DecisionTreeClassifier()
Model_DT.fit(X_train,y_train)

Y_pred = Model_DT.predict(X_test)
DT_param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the GridSearchCV object
DT_grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=DT_param_grid, cv=5, scoring='accuracy', verbose=2)

# Fit the model
DT_grid_search.fit(X_train, y_train)

y_pred_DT = DT_grid_search.predict(X_test)
accuracy_DT = accuracy_score(y_test,y_pred_DT)
Confusion_DT = confusion_matrix(y_test,y_pred_DT)


# %%
print("Decision Tree - Best Params:", DT_grid_search.best_params_)
print("Decision Tree - Best Estimator:", DT_grid_search.best_estimator_)
print("Decision Tree - Best Score:", DT_grid_search.best_score_)
print("Accuracy:",accuracy_DT)
print(Confusion_DT)
Plot_5(y_test,y_pred_DT)

# %% [markdown]
# ##  Random Forest

# %%
Model_RF = RandomForestClassifier()
Model_RF.fit(X_train,y_train)


RF_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features': [ 'sqrt', 'log2']   
}

RF_grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=RF_param_grid, scoring = 'accuracy',cv=5,verbose= 2)
RF_grid_search.fit(X_train, y_train)


y_pred_RF = RF_grid_search.predict(X_test)
accuracy_RF = accuracy_score(y_test,y_pred_RF)
Confusion_RF = confusion_matrix(y_test,y_pred_RF)

# %%
print("Accuracy:",accuracy_RF)
print(Confusion_RF)

print("Random Forest - Best Params:", RF_grid_search.best_params_)
print("Random Forest - Best Estimator:",RF_grid_search.best_estimator_)
print("Random Forest - Best Score:",RF_grid_search.best_score_)
Plot_5(y_test,y_pred_RF)

# %% [markdown]
# ### SVM

# %%
Model_SVM = SVC()
Model_SVM.fit(X_train, y_train)

svc_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4],
    'coef0': [0.0, 0.1, 0.5]
}

svc_grid_search = GridSearchCV(estimator=SVC(), param_grid=svc_param_grid, cv=2,verbose=2,scoring="accuracy")
history = svc_grid_search.fit(X_train, y_train)

y_pred_SVM = svc_grid_search.predict(X_test)
accuracy_SVM = accuracy_score(y_test, y_pred_SVM)

# %%
print("Accuracy:", accuracy_SVM)
print("SVM - Best Params:", svc_grid_search.best_params_)
print("SVM - Best Estimator:", svc_grid_search.best_estimator_)
print("SVM - Best Score:", svc_grid_search.best_score_)
Plot_5(y_test, y_pred_SVM)


