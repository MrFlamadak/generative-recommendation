import encodings
import string
from os import write

import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from pandas import DataFrame

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
def create_and_pickle_user_profiles():

    #Creates a user profile for each customer containing article ids of all articles that have been bought.
    transactions_df = pd.read_pickle("transactions_train.pkl")
    user_profiles_df = transactions_df.groupby("customer_id", as_index=False)["article_id"].agg(list)

    user_profiles_df.to_pickle("user_profiles.pkl")
    return user_profiles_df


def create_test_and_training_user_profiles(frac):
    user_profiles_df = pd.read_pickle("user_profiles.pkl")
    seed = 42
    user_profiles_shuffled_df = user_profiles_df.sample(frac=1, random_state=seed)

    partition_index = int(frac * len(user_profiles_df))
    train_user_profiles_df = user_profiles_shuffled_df[0:partition_index]
    test_user_profiles_df = user_profiles_shuffled_df[partition_index:]

    print(f"Percentage of data in train: {len(train_user_profiles_df) / len(user_profiles_df)}")
    print(f"Percentage of data in test: {len(test_user_profiles_df) / len(user_profiles_df)}")
    return train_user_profiles_df, test_user_profiles_df

def info_about_article_dataset():
    article_df = pd.read_pickle("articles.pkl")

    # Data analysis (to understand the data)
    print(f"Dimensions, column names and datatypes of the data:\n")
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
    return