import encodings
import string
from os import write

import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from pandas import DataFrame

# Code that read from original csv and converted to very fast format pkl
'''
customer_df = pd.read_csv("customers.csv")
transactions_df = pd.read_csv("transactions_train.csv")
article_df = pd.read_csv("articles.csv")

article_df.to_pickle("articles.pkl")
customer_df.to_pickle("customers.pkl")
transactions_df.to_pickle("transactions_train.pkl")

'''
def get_article_feature_string_list():
    article_df = pd.read_pickle("articles.pkl")

    article_df_no_numbers = article_df[["article_id", "graphical_appearance_name", "perceived_colour_value_name",
                                        "perceived_colour_master_name", "prod_name", "detail_desc",
                                        "product_type_name", "product_group_name", "department_name",
                                        "index_name", "index_group_name", "section_name", "garment_group_name"]]

    article_df_strings = pd.DataFrame({
        "feature_string": article_df_no_numbers.astype(str).agg(" ".join, axis=1)
    })

    return article_df_strings.values.tolist()

def get_user_profile_list():
    return pd.read_pickle("user_profiles.pkl")

def create_test_and_training_user_profiles(frac):
    customer_df = pd.read_pickle("customers.pkl")
    seed = 42
    customer_shuffled_df = customer_df.sample(frac=1, random_state=seed)

    partition_index = int(frac * len(customer_df))
    train_customer_df = customer_shuffled_df[0:partition_index]
    test_customer_df = customer_shuffled_df[partition_index:]

    print(f"Percentage of data in train: {len(train_customer_df) / len(customer_df)}")
    print(f"Percentage of data in test: {len(test_customer_df) / len(customer_df)}")
    return train_customer_df, test_customer_df



#What are good functions to include? get customer data, get article dataa etc
#Remove duplicate words

# Read Data
customer_df = pd.read_pickle("customers.pkl")

article_feature_strings = get_article_feature_string_list()

article_df = pd.read_pickle("articles.pkl")
#transactions_df = pd.read_pickle("transactions_train.pkl")

print("Example of article row")
#print(article_df.iloc[3])


#print(article_df_no_numbers.iloc[123, 12])

# Creates a dataframe with concatenated columns(1 string per row) (uses vectorized operation (very fast))
# article_id    feature_string
#   ...             ...
#   ...             ...

print(article_feature_strings.iloc[1])

# Split customers into 80/20, training and test sets
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