################################
# FLO Customer Segmentation with Unsupervised Machine Learning
################################

#### PROJECT DEFINITION
# FLO wants to segment customers in order to define new marketing strategies.
# The consumption behaviours of individuals are defined and clusters / customer groups are formed by unsupervised machine learning techniques
# RFM Analysis is used for feature engineering

#The dataset includes Flo's last purchases from OmniChannel (both online and offline shoppers) in 2020 - 2021.
#It consists of information obtained from the past shopping behaviors of customers


# Features
#
# master_id: Unique customer ID
# order_channel: The channel used (Android, ios, Desktop, Mobile) for transaction
# last_order_channel: Last channel used for shopping
# first_order_date: First date the customer shopped
# last_order_date: Last date the customer shopped
# last_order_date_online: Last date the customer shopped in the online channel
# last_order_date_offline: Last date the customer shopped in the offline channel
# order_num_total_ever_online: Total number of transactions on the online channel
# order_num_total_ever_offline: Total number of transactions on the offline channel
# customer_value_total_ever_offline: Total amount of transaction in offline platforms
# customer_value_total_ever_online: Total amount of transaction in online platforms
# interested_in_categories_12: Categories customer shopped in the last 12 months

################################
# Import Libraries
################################

import numpy as np
import pandas as pd
import datetime as dt
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import warnings


# import helper.py
import helper

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

################################
# EDA
################################

# Step 1: Data Preparation

df_ = pd.read_csv("data/raw/flo_data_20k.csv")
df = df_.copy()
df.head(5)

#Step 2: Feature Engineering using RFM Analysis

df["customer_value_total_ever_omnichannel"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df["order_num_total_ever_omnichannel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

df.head()

df.info()

df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)

rfm = df.groupby("master_id").agg({"last_order_date": lambda last_order_date: (today_date - last_order_date.max()).days,
                                   "order_num_total_ever_omnichannel": lambda frequency: frequency.sum(),
                                   "customer_value_total_ever_omnichannel": lambda total: total.sum()})

#other methods -- rfm["recency"] = (today_date - df["last_order_date"]).astype("timedelta64[0])
rfm.columns = ["recency", "frequency", "monetary"]
#rfm["recency"] = (today_date - rfm["last_order_date"])

rfm = rfm.reset_index()
df.head(5)

df_merged = pd.merge(df,rfm)

df_merged.head()
df_merged.info()

final_df = df_merged[["order_channel", "last_order_channel","recency","frequency", "monetary"]]
#helper.correlation_matrix(final_df, num_cols)

cat_cols, num_cols, cat_but_car = helper.grab_col_names(final_df)

df = helper.one_hot_encoder(final_df, cat_cols, drop_first=True)
df.head()

################################
# K-Means Clustering
################################

#Step 3: Standardization
s = StandardScaler()
X_scaled = s.fit_transform(df[num_cols])
df[num_cols] =pd.DataFrame(X_scaled, columns=df[num_cols].columns)
df.head()

#Step 4: Finding the optimum number of clusters
kmeans = KMeans()
ssd = []
K = range(1,30)


for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

#len(ssd)

plt.plot(K, ssd, "bx-")
plt.xlabel("SSE/SSR/SSD for different K values")
plt.title("Elbow Method to find the Optimum Cluster Number")
plt.show()

kmeans= KMeans()
elbow= KElbowVisualizer(kmeans, k=(2,20))
elbow.fit(df)
elbow.show()

#the optimum K
elbow.elbow_value_

#Step 5: Machine Learning Model
# Final Clusters

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

cluster_kmeans = kmeans.labels_
df.head()

final_df["kmeans_clusters"] = cluster_kmeans + 1


# Step 6: Statistical Analysis
final_df.groupby("kmeans_clusters").describe()

################################
# Hierarchical Clustering
################################
hc_average = linkage(df, "average")




# more comprehensive plot??
# plt.figure(figsize=(10,5))
# plt.title("Hierarchical Dendogram")
# plt.xlabel("Observations")
# plt.ylabel("Distances")
# dendrogram(hc_average, leaf_font_size=10)
# plt.show()

plt.figure(figsize=(7, 10))
plt.title("Hierarchical Dendrogram")
plt.xlabel("Observations")
plt.ylabel("Distances")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=20, color='r', linestyle='--')
plt.axhline(y=25, color='b', linestyle='--')
plt.show()

#FINAL MODEL
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=3, linkage="average")
clusters = cluster.fit_predict(df)

final_df.head()

final_df["hi_clusters"] = clusters + 1

final_df.groupby("hi_clusters").describe()
