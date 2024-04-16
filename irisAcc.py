import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the Iris dataset
df = pd.read_csv('iris_data.csv')

# Print the full dataset
print(df.to_string())

# Plot the dataset
df.plot()
plt.show()

# Select features and labels
features = df[['petal length', 'petal width', 'sepal length', 'sepal width']]
labels = df['class']

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(train_features, train_labels)

# Get the coefficients (weights) and intercept
weights = model.coef_
intercept = model.intercept_

print("weights: ", weights)
print("intercept: ", intercept)

# Make predictions on the test set
predictions = model.predict(test_features)

# Evaluate the accuracy of the predictions
accuracy = model.score(test_features, test_labels)
print("Accuracy:", accuracy)

#-----------------------------------------------------------------------------------------------------------------------------#
#THIS IS A CLASSIFICATION PROBLEM, SO HERE IT IS BY KNN(K-NEAREST NEIGHBOUR)

features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=42)

scaler=StandardScaler()
features_train_scaled=scaler.fit_transform(features_train)
features_test_scaled=scaler.transform(features_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(features_train, labels_train)
labels_pred = knn.predict(features_test)

accuracy = accuracy_score(labels_test, labels_pred)
print("Accuracy:", accuracy)
