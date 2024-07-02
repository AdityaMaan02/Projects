import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sb

df=pd.read_csv("netflix_titles.csv", encoding='latin-1')

print(df.info())

# type encoded
label_encoder=LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])

# title encoded
df['title'] = label_encoder.fit_transform(df['title'])
print(df["title"].value_counts())

# director encoded
df['director'] = label_encoder.fit_transform(df['director'])
df['director'] = df['director'].fillna(df['director'].median())
print(df["director"].value_counts())
print(df["director"].nunique())

# country encoded
top_countries = df['country'].value_counts().head(10).index.tolist()
country_label_map = {country: label for label, country in enumerate(top_countries, start=1)}
df['country'] = df['country'].map(country_label_map).fillna(0)
df['country'] = df['country'].astype(int)
print(df['country'].value_counts())

# release_year encoded
release_year_list = df['release_year'].value_counts()
top_release_years = release_year_list.index[:8].tolist()
df['release_year'] = df['release_year'].apply(lambda x: int(x) if x in top_release_years else 0)
print(df['release_year'].value_counts())

# rating encoded
df['rating'] = df['rating'].fillna(0)
top_ratings = df['rating'].value_counts().index[:9].tolist()
df['rating'] = df['rating'].apply(lambda x: x if x in top_ratings else '0')
df['rating'] = label_encoder.fit_transform(df['rating'])
print(df["rating"].value_counts())

# duration encoded
df['duration'] = df['duration'].fillna(0)
top_durations = df['duration'].value_counts().index[:12].tolist()
df['duration'] = df['duration'].apply(lambda x: x if x in top_durations else '0')
df['duration'] = label_encoder.fit_transform(df['duration'])
print(df['duration'].value_counts())

# listed_in encoded
df['listed_in'] = df['listed_in'].fillna(0)
top_categories = df['listed_in'].value_counts().index[:12].tolist()
df['listed_in'] = df['listed_in'].apply(lambda x: x if x in top_categories else '0')
df['listed_in'] = label_encoder.fit_transform(df['listed_in'])
print(df['listed_in'].value_counts())

df.drop(["show_id","description","cast","date_added"], axis=1, inplace=True)
df = df.dropna(axis=1, how='all')

print(df.info())

# feeding data to DBSCAN model
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels = dbscan.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', marker='o', s=50)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("DBSCAN Clustering")
plt.show()