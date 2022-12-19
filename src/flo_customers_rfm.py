### Flo Customer Segmentation with RFM Analysis ###

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
#packages
import pandas as pd
import datetime as dt
import sys
import os

module_path = os.path.abspath(os.getcwd())

if module_path not in sys.path:
    sys.path.append(module_path)

import helper

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

################################
# EDA
################################

#### STEP 1: Data Preparation & Exploratory Data Analysis ####
# Read Data
df_ = pd.read_csv("data/raw/flo_data_20k.csv")
df = df_.copy()

#observations
helper.check_df(df)


df["master_id"].nunique()

################################
# FEATURE ENGINEERING
################################

#new varibles
# the total number of purchases and spending in both online and offline channels of each customer are defined as new features

df["customer_value_total_ever_omnichannel"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df["order_num_total_ever_omnichannel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df.head()

#datetime
# converting date to datetime format

df.info()
df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])

#another-simpler method
# date_columns = df.columns[df.columns.str.contains("date")]
#df[date_columns] = df[date_columns].apply(pd.to_datetime)


df.head()

# See the distribution of the number of customers in the shopping channels, the total number of products purchased and the total expenditures.
df.groupby("order_channel").agg({"master_id": "count",
                                 "order_num_total_ever_omnichannel" : "sum",
                                 "customer_value_total_ever_omnichannel" : "sum"
                                 })

# top 10 customers with the highest revenue
df[["master_id", "customer_value_total_ever_omnichannel"]].sort_values("customer_value_total_ever_omnichannel", ascending=False).head(10) #[:10]

# top 10 customers with most orders.
df.sort_values("order_num_total_ever_omnichannel", ascending=False).head(10)


# Function for data preparation
def prepare_data(dataframe):
    dataframe["customer_value_total_ever_omnichannel"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe["order_num_total_ever_omnichannel"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["first_order_date"] = pd.to_datetime(dataframe["first_order_date"])
    dataframe["last_order_date"] = pd.to_datetime(df["last_order_date"])
    dataframe["last_order_date_online"] = pd.to_datetime(dataframe["last_order_date_online"])
    dataframe["last_order_date_offline"] = pd.to_datetime(dataframe["last_order_date_offline"])

    return dataframe


df = df_.copy()
df.head()

new_df = prepare_data(df)
new_df.head()
new_df.info()



################################
# RFM METRICS
################################

#Step 1: Definition of Recency, Frequency and Monetary
#Step 2: Calculate the Recency, Frequency and Monetary metrics specific to the customer.
#Step 3: Assign your calculated metrics to a variable named rfm.
#Step 4: Rename metrics to recency, frequency and monetary olarak değiştiriniz.
df = new_df.copy()
df.head()

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)
type(today_date)

rfm = df.groupby("master_id").agg({"last_order_date": lambda last_order_date: (today_date - last_order_date.max()).days,
                                   "order_num_total_ever_omnichannel": lambda frequency: frequency.sum(),
                                   "customer_value_total_ever_omnichannel": lambda total: total.sum()})

#another method -- rfm["recency"] = (today_date - df["last_order_date"]).astype("timedelta64[0])
rfm.columns = ["recency", "frequency", "monetary"]
#rfm["recency"] = (today_date - rfm["last_order_date"])

rfm.head()
rfm.info()
rfm.describe().T
rfm = rfm.reset_index()

#rfm[rfm["master_id"] == "cc294636-19f0-11eb-8d74-000d3a38a36f"]
rfm.columns


################################
# RF Scores
################################
# Step 1: Redefine Recency, Frequency and Monetary metrics with qcut into scores between 1-5
# Step 2: Save these as recency_score, frequency_score and monetary_score
# Step 3: Define recency_score and frequency_score as one feature named RF_SCORE

rfm.head()

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1])

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])

rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1,2,3,4,5])

#convert scores to string format
rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

rfm.head()
rfm.describe().T

rfm[rfm["RF_SCORE"] == "55"]


################################
# RF Scores as Customer Segment
################################

# Step 1: segment definitions for the generated RF scores
# Step 2: Convert scores into segments with the help of seg_map.

#regex
seg_map = {
    r'[1-2][1-2]' : "hibernating",
    r'[1-2][3-4]' : "at_Risk",
    r'[1-2]5' : "cant_loose",
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
rfm.head()


# Examine the recency, frequency and monetary averages of the segments.
# With the help of RFM analysis, find the customers in the relevant profile for the 2 cases given below and save the customer IDs as csv.

# FLO has a new women's shoe brand. The product prices of the brand are above the general customer preferences.
# For this reason, it is desired to contact the customers in the profile that will be interested in promoting the brand and product sales.
# Customers to be contacted privately: from loyal customers (champions, loyal_customers) and women.
# Save the id numbers of these customers to the csv file.


#rfm.groupby("segment"). agg(["mean", "count"])
rfm[["segment","recency", "frequency", "monetary"]].groupby("segment"). agg(["mean", "count"])

df.head()
interests = df[["master_id", "interested_in_categories_12"]]

interests.head()

flo_new = rfm.merge(interests, on="master_id", how="right")
flo_new.head()

loyals = flo_new[flo_new["segment"].isin(["loyal_customers", "champions"])]
#loyals = flo_new[(flo_new["segment"] == "loyal_customers") | (flo_new["segment"] == "champions")]
loyals.head()

loyals["interested_in_categories_12"].unique()
#loyals.loc[:, ~loyals["interested_in_categories_12"].str.contains("KADIN")].head()

woman = loyals[loyals["interested_in_categories_12"].str.contains("KADIN")]
woman.info()
woman.head(10)
woman.shape
loyals.shape

#loyals.info()
#loyals.head()

woman.to_csv("loyal_womans.csv")

# b. Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır.
# Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler,
# uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.

# Nearly 40% discount is planned for Men's and Children's products.
# Customers who were good customers in the past, but have not shopped for a long time and potentially, interested in the categories related to this discount,
# dormant and newly arriving customers are specifically targeted.
# Save the ids of the customers in the appropriate profile to the csv file.

rfm["segment"].unique()

# "loyal_customers", "need_attention", "new_customers", "about_to_sleep"
target = flo_new[flo_new["segment"]. isin(["loyal_customers", "need_attention", "new_customers", "about_to_sleep"])]

target.info()
target["segment"].unique()
new_target = target[target["interested_in_categories_12"].str.contains("ERKEK|COCUK")]
#target["interested_in_categories_12"].unique()

new_target.head()
new_target.to_csv("target_discount.csv")





