import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC                             #import SVR if regression OR import SVC if classification
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sb

df=pd.read_csv("onlinefoods.csv")

# df.describe()

#Gender cleaned
df['Gender'] = df['Gender'].replace('Male', 0)
df['Gender'] = df['Gender'].replace('Female', 1)
print(df['Gender'].value_counts())

#Marital Status cleaned
df['Marital Status'] = df['Marital Status'].replace('Single', 0)
df['Marital Status'] = df['Marital Status'].replace('Married', 1)
df['Marital Status'] = df['Marital Status'].replace('Prefer not to say', 2)
print(df['Marital Status'].value_counts())

#Occupation cleaned
df['Occupation'] = df['Occupation'].replace('Student', 0)
df['Occupation'] = df['Occupation'].replace('Employee', 1)
df['Occupation'] = df['Occupation'].replace('Self Employeed', 2)
df['Occupation'] = df['Occupation'].replace('House wife', 3)
print(df['Occupation'].value_counts())

#Monthly Income cleaned
label_encoder = LabelEncoder()
df['Monthly Income'] = label_encoder.fit_transform(df['Monthly Income'])
print(df['Monthly Income'].value_counts())

#Educational Qualifications cleaned
df['Educational Qualifications'] = label_encoder.fit_transform(df['Educational Qualifications'])
print(df['Educational Qualifications'].value_counts())

#Output cleaned
df['Output'] = label_encoder.fit_transform(df['Output'])
print(df['Output'].value_counts())

#Feedback cleaned
df['Feedback'] = label_encoder.fit_transform(df['Feedback'])
print(df['Feedback'].value_counts())

#Unnamed column dropped
df.drop(df.columns[df.columns.str.contains('Unnamed', case=False)], axis=1, inplace=True)
df.info()

#SVC model
X=df.drop("Output",axis=1)
Y=df["Output"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
svc=SVC()
svc.fit(X_train,Y_train)

Y_pred=svc.predict(X_test)

accuracy=accuracy_score(Y_test,Y_pred)
print("Accuracy: ",accuracy)
