import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import seaborn as sb

df=pd.read_csv('Indian_housing_Delhi_data.csv')
#print(df.to_string())
print(len(df))
df.info()
df.describe()



#data cleaning
df.drop(['isNegotiable','verificationDate','description','currency','latitude','longitude','city'], axis=1, inplace=True)   #drops these columns
df['SecurityDeposit'] = df['SecurityDeposit'].replace('No Deposit', '0')           #replace
# df['SecurityDeposit_int'] = df['SecurityDeposit'].apply(lambda x: int(x.replace(',', '').strip()))                  #replaces ',' and converts to integer
# df['SecurityDeposit']=df['SecurityDeposit'].replace(',', '')
df['SecurityDeposit'] = df['SecurityDeposit'].replace(',', '', regex=True).astype(float)


df['Status'] = df['Status'].replace('Unfurnished', 0)
df['Status'] = df['Status'].replace('Semi-Furnished', 1)
df['Status'] = df['Status'].replace('Furnished', 2)


print(df['house_type'].unique())                                                   #shows all unique values
house_type_list=df['house_type'].value_counts()                                    #counts all unique value's frequency
print(house_type_list)

#encoding house_type
house_type_list=house_type_list.iloc[:3]                                #encode from 0-3(4 values)
label_encoder = LabelEncoder()
df['house_type']=df['house_type'].apply(lambda x: x if x in house_type_list else 'other')
df['house_type_encoded'] = label_encoder.fit_transform(df['house_type'])
df.drop(['house_type'], axis=1, inplace=True)


print(df['numBalconies'].isnull().sum())                                #counts frequency of all null values
print(df['numBalconies'].unique())                                      #shows all unique values 

#encoding numBalconies
numBalconies_list=df['numBalconies'].value_counts()                                    #counts all unique value's frequency
print(numBalconies_list)
numBalconies_list=numBalconies_list.iloc[:3]
df['numBalconies']=df['numBalconies'].apply(lambda x: int(x) if x in numBalconies_list else 0)

#encoding location
location_list=df['location'].value_counts()
location_list=location_list.iloc[:20]
df['location']=df['location'].apply(lambda x: x if x in location_list else 'other')
df['location_encoded'] = label_encoder.fit_transform(df['location'])
df.drop(['location'], axis=1, inplace=True)

#treating nan values in numBathrooms
mean_num_bathrooms = df['numBathrooms'].mean()
df['numBathrooms'] = df['numBathrooms'].fillna(mean_num_bathrooms)


print(df['house_type_encoded'].value_counts())  
print(df['location_encoded'].value_counts())  
print(df['numBalconies'].value_counts())

def convert_to_int(area_str):
    area_numeric_str = ''.join(filter(str.isdigit, area_str))
    return int(area_numeric_str)
df['house_size_int'] = df['house_size'].apply(convert_to_int)
df.drop(['house_size'], axis=1, inplace=True)

df['priceSqFt']=df['price']/df['house_size_int']

print(df['priceSqFt'])
#print(df['house_type_encoded'].value_counts())

# print(df.corr())
# plt.show()

df.info()
print(df['price'].isnull().sum())
print(df['numBathrooms'].isnull().sum())
print(df['numBalconies'].isnull().sum())
print(df['priceSqFt'].isnull().sum())
print(df['SecurityDeposit'].isnull().sum())
print(df['Status'].isnull().sum())
print(df['house_type_encoded'].isnull().sum())
print(df['location_encoded'].isnull().sum())
print(df['house_size_int'].isnull().sum())




X=df.drop('price',axis=1)
Y=df['price']

#MATPLOTLIB
plt.scatter(df['house_type_encoded'],Y)
plt.xlabel("features")
plt.ylabel("price")
plt.title("scatter plot")
plt.show()

#SEABORN
sb.pairplot(df, x_vars=['house_size_int', 'numBathrooms', 'numBalconies','house_type_encoded','location_encoded','SecurityDeposit','Status'], y_vars=['price'])

#sb.pairplot(df, x_vars=X.columns, y_vars=Y)                       DOESN'T WORK, AS SEABORN'S "x/y_vars" EXPECTS COLUMN NAMES OR ARRAYS-LIKE OBJECTS AND NOT ENTIRE DATA FRAMES (X or Y).
plt.show()


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

#-----------------------------------------------------------------------------------------------------------------------------#
#THIS DATA IS A REGRESSION PROBLEM, SO HERE IT IS BY LINEAR REGRESSION

model = LinearRegression()
model.fit(X_train, Y_train)

# Get the coefficients (weights) and intercept
weights = model.coef_
intercept = model.intercept_

print("weights: ", weights)
print("intercept: ", intercept)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the accuracy of the predictions
accuracy = model.score(X_test, Y_test)
print("Accuracy:", accuracy)