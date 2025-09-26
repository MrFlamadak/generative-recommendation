import encodings
import string
from os import write

import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

# Code that read from original csv and converted to very fast format pkl
'''
customer_df = pd.read_csv("customers.csv")
transactions_df = pd.read_csv("transactions_train.csv")
article_df = pd.read_csv("articles.csv")

article_df.to_pickle("articles.pkl")
customer_df.to_pickle("customers.pkl")
transactions_df.to_pickle("transactions_train.pkl")

'''
# Read Data
customer_df = pd.read_pickle("customers.pkl")
article_df = pd.read_pickle("articles.pkl")
#transactions_df = pd.read_pickle("transactions_train.pkl")


# Creates a dataframe with concatenated columns(1 string per row) (uses vectorized operation (very fast))
article_df_strings = article_df.astype(str).agg(" ".join, axis=1)

# Split customers into 80/20, training and test
seed = 42
customer_shuffled_df = customer_df.sample(frac=1, random_state=seed)

partition_index = int(0.8*len(customer_df))
train_customer_df = customer_shuffled_df[0:partition_index]
test_customer_df = customer_shuffled_df[partition_index:]

print(f"Percentage of data in train: {len(train_customer_df) / len(customer_df)}")
print(f"Percentage of data in test: {len(test_customer_df) / len(customer_df)}")


# Create user items list each customer should have a list of article ids which he has bought.

#customer_item_list_df = transactions_df.groupby("customer_id", as_index=False)["article_id"].agg(list)
# Create pickle file because very slow

#customer_item_list_df.to_pickle("user_profiles.pkl")
user_profiles_df = pd.read_pickle("user_profiles.pkl")

print(user_profiles_df.iloc[0])

# Data analysis (to understand the data)
print(f"Shows dimensions, column names and datatypes of the data:\n")
print(article_df.info()) # Seems to be a mix of int64 and objects(strings)

print("Counts number of unique values for each column:\n")
print(article_df.nunique()) # Contains many duplicated information columns, see ..._no and ..._name.

# TAKES WAY TOO LONG TIME TO DO, WE SKIP FOR NOW
# print("Calculates correlation between two columns in articles:\n")
#one_hot_article_df_pdn = pd.get_dummies(article_df["prod_name"])
#print(one_hot_article_df_pdn.corrwith(article_df["product_code"]))

# Creates a barchart showing distribution of articles with each respective section name
counts = article_df["section_name"].value_counts()
print(counts)
counts.plot(figsize=(12,7), kind="bar")
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.xlabel("Section")
plt.ylabel("Count")
plt.title("Count of articles belonging to each respective section")
plt.show()
# 1/3 of all articles are concentrated in two sections and 18 sections have less than 1000 articles each
# with bottom 4 having less than 50