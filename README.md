import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score
data = pd.read_csv("weather_classification_data.csv")
data
print(data.isna().sum())
print(data.duplicated().sum())
print(data.info())


print(data["Weather Type"].unique())
print(data.describe()) 

# data.columns
new_order  = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index','Visibility (km)','Cloud Cover', 'Season', 'Location', 'Weather Type']
data = data[new_order]

data

def Plot_1(data, col):
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(data[col], bins=10, edgecolor='black',color= 'skyblue')
    plt.title(f'{col} Distribution')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(data[col])
    plt.title(f'{col} BoxPlot')
    
    plt.tight_layout()
    plt.show()
    
for i in range(4):
    Plot_1(data, data.columns[i])

def Plot_2(data, col):

    palette = sns.color_palette("Set2", len(data[col].unique()))
    count = sns.countplot(data=data, x=col, order=data[col].value_counts().index,palette= palette)
    total = len(data)
    
    for p in count.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        count.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')

    plt.show()

for i in range(-4,-1):
    Plot_2(data,data.columns[i])
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

# Example usage
Plot_3(data, 'Temperature')

sns.pairplot(data = data,
             hue = "Weather Type")
data_1 = data

data_1['Cloud Cover'] = pd.factorize(data_1["Cloud Cover"])[0] + 1
data_1['Season'] = pd.factorize(data_1['Season'])[0] +1
data_1['Location'] = pd.factorize(data_1['Location'])[0] +1
data_1['Weather Type'] = pd.factorize(data_1["Weather Type"])[0] +1

data_1
correlation= data_1.iloc[:,0:10].corr()
fig = px.imshow(correlation,
                text_auto = True,
                aspect = "auto",
                color_continuous_scale='YlOrRd',
                title='Correlation Matrix')

fig.update_layout(xaxis = dict(tickangle = -45),width = 1000,height = 700)
fig.show()
### Train test split operation 
X = data_1.iloc[:,0:10]
y = data_1.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state= 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

### Scaling the splitted dataset 

Scalar = MinMaxScaler()
X_train = Scalar.fit_transform(X_train)
X_test = Scalar.fit_transform(X_test)
### LogisticRegression
Model_LR = LogisticRegression()
Model_LR.fit(X_train,y_train)

y_pred_LR = Model_LR.predict(X_test)
accuracy = accuracy_score(y_test,y_pred_LR)
Confusion_LR = confusion_matrix(y_test,y_pred_LR)

print(Confusion_LR)
print(accuracy)


### Decision Tree
Model_DT = DecisionTreeClassifier()
Model_DT.fit(X_train,y_train)

y_pred_DT = Model_DT.predict(X_test)
accuracy_DT = accuracy_score(y_test,y_pred_DT)
Confusion_DT = confusion_matrix(y_test,y_pred_DT)

print(accuracy_DT)
print(Confusion_DT)
###  Random Forest
Model_RF = RandomForestClassifier()
Model_RF.fit(X_train,y_train)

y_pred_RF = Model_RF.predict(X_test)
accuracy_RF = accuracy_score(y_test,y_pred_RF)
Confusion_RF = confusion_matrix(y_test,y_pred_RF)

print(accuracy_RF)
print(Confusion_RF)
plt.figure(figsize=(8, 6))
sns.heatmap(Confusion_RF, annot=True, fmt='d', cmap='Blues', xticklabels=data yticklabels=data)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
