# churnrate_telecom_bank

# chrun_rate_telecom

Minimizing Churn Rate Through Analysis of Financial Habits

# Table of Content
1. [Project Overview](#project)
2. [Dataset Overview](#dataset)
3. [Steps](#steps)
4. [Model Choose](#model)

<a name="project"></a>
## Project Overview

"Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." [IBM Sample Data Sets]

<a name="dataset"></a>
## Dataset overview
Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

The data set includes information about:

Customers who left within the last month – the column is called Churn
Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
Demographic info about customers – gender, age range, and if they have partners and dependents
<a href='https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbEh4OFcwTkpjSW8xZE90RnhzUWlKNE1PZDhXQXxBQ3Jtc0tuUjdoeHZGTW9sUWJvaHYxWjZpUmJ4cWdwUGJPQjJGQndIYzU5MUNRYnh0TmVCNExtWWQxODQxbEx1b0RDMUVVV0d5d3dQcDdGOWhsZjlScml1aTB5ZTVXYmRSUFBSREswMGRmRVFGV0FRNWdHeTFFbw&q=https%3A%2F%2Fwww.kaggle.com%2Fblastchar%2Ftelco-customer-churn'>Dataset Link</a>
<a name="steps"></a>
## Steps  
1. Access, Clean and Analyze Data
gender               object
SeniorCitizen         int64
Partner              object
Dependents           object
tenure                int64
PhoneService         object
MultipleLines        object
InternetService      object
OnlineSecurity       object
OnlineBackup         object
DeviceProtection     object
TechSupport          object
StreamingTV          object
StreamingMovies      object
Contract             object
PaperlessBilling     object
PaymentMethod        object
MonthlyCharges      float64
TotalCharges         object
Churn                object
dtype: object

Above chart you can see total charges are object not float
so we need to convert string to numbers
```
df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]
```
Replace yes to no
```
yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df1[col].replace({'Yes': 1,'No': 0},inplace=True)
```
 
One hot encoding
```
df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'])
df2.columns
```
<a name="model"></a>
## Model Choose

I tried ANN 
#### Model Building ####

```
import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)
```


# chrun_rate_bank

# Table of Content
1. [Project Overview](#project1)
2. [Business Challenge](#Business_Challenge1)
3. [Dataset Overview](#dataset1)
4. [Steps](#steps1)
5. [Model Choose](#model1)

<a name="project1"></a>
## Project Overview

Given a Bank customer, can we build a classifier which can determine whether they will leave in the next 6 months or not?


## Dataset overview


<a name="steps1"></a>
## Steps  
1. Access, Clean and Analyze Data
```
#Converting Gender into binary
df['Gender'].replace({'Female': 0, 'Male': 1}, inplace= True)

df['Geography'].replace({'France': 0, 'Spain': 1,'Germany':2}, inplace= True)

```
Minmax scalar
```
c_to_scale= ['CreditScore','Age','Balance','EstimatedSalary','Tenure']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[c_to_scale] = scaler.fit_transform(df[c_to_scale])
```

<a name="model1"></a>
## Model Choose

I tried logistic regression model to check which model 
#### Model Building ####

```
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0,class_weight='Balanced',C=0.1)
```


