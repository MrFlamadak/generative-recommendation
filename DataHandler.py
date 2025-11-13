import encodings
import random
import string
from os import write

import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from pandas import DataFrame

import data_analyzer


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
    transactions_df = pd.read_pickle("transactions_train.pkl")
    user_profiles_df = transactions_df.groupby("customer_id", as_index=False)["article_id"].agg(list)

    user_profiles_df.to_pickle("user_profiles.pkl")
    return user_profiles_df


def create_train_val_test_user_profiles():
    user_profiles_df = pd.read_pickle("user_profiles.pkl")
    user_profiles_df = remove_under_threshold(user_profiles_df, 2)
    seed = 42
    user_profiles_shuffled_df = user_profiles_df.sample(frac=1, random_state=seed)

    partition_index_1 = int(0.6 * len(user_profiles_df))
    partition_index_2 = int(0.8 * len(user_profiles_df))
    train_user_profiles_df = user_profiles_shuffled_df[0:partition_index_1]
    val_user_profiles_df = user_profiles_shuffled_df[partition_index_1:partition_index_2]
    test_user_profiles_df = user_profiles_shuffled_df[partition_index_2:]

    print(f"Percentage of data in train: {len(train_user_profiles_df) / len(user_profiles_df)}")
    print(f"Percentage of data in validation: {len(val_user_profiles_df) / len(user_profiles_df)}")
    print(f"Percentage of data in test: {len(test_user_profiles_df) / len(user_profiles_df)}")

    train_user_profiles_df.to_pickle("customer_transactions_TRAIN60P.pkl")
    val_user_profiles_df.to_pickle("customer_transactions_VAL20P.pkl")
    test_user_profiles_df.to_pickle("customer_transactions_TEST20P.pkl")

    return train_user_profiles_df, val_user_profiles_df, test_user_profiles_df

def get_random_item_to_sem_ids(size):
    article_data = pd.read_pickle("articles.pkl")
    item_to_sem = pd.read_pickle("item_2_semantic.pkl")

    random_item_to_sem_ids = random.sample(list(item_to_sem.items()), size)
    chosen_sem_ids = [sem_id for _, sem_id in random_item_to_sem_ids]
    chosen_sem_ids = np.array(chosen_sem_ids)

    chosen_article_ids = [int(article_id) for article_id, _ in random_item_to_sem_ids]
    chosen_article_data = article_data[article_data["article_id"].isin(chosen_article_ids)]
    # Reorders so that indices match
    chosen_article_data = article_data.set_index("article_id").loc[chosen_article_ids].reset_index()

    return chosen_sem_ids, chosen_article_data

def remove_under_threshold(user_profile, threshold):
    """
    Removes all rows from user_profile where the number of articles
    is less than the given threshold.

    Parameters
    ----------
    user_profile : pd.DataFrame
        DataFrame with columns ['customer_id', 'article_id'], where 'article_id' is a list.
    threshold : int
        Minimum number of articles a customer must have to be kept.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only customers with article counts >= threshold.
    """
    # Compute number of articles per customer
    user_profile['num_articles'] = user_profile['article_id'].apply(len)

    # Filter rows based on threshold
    preprocessed_user_profile = user_profile[user_profile['num_articles'] >= threshold].copy()

    # Optionally drop helper column if not needed
    preprocessed_user_profile.drop(columns='num_articles', inplace=True)

    return preprocessed_user_profile

def split_input_label_transactions(transactions, input_size=3, labels=1):
    """
    Splits the transaction lists into inputs and labels. Intended for
    the test data.
    
    Parameters
    ----------
    transactions : dict[string -> list[int]]
        Dictionary with key-value pairs ('customer_id', semantic_id_list)
    input_size : int
        Defines the amount of inputs needed from each user's transaction list.
    labels : int
        Defines the amount of labels needed from each user's transaction list.
        
    Returns
    -------
    split_transactions : dict[string -> (list[int], list[int])]
        Dictionary with key-value pairs ('customer_id', (input_semantic_id_list, label_semantic_id_list))
    """
    
    split_transactions = {}
    
    for key, list in transactions.items():
        split_transactions[key] = (list[:input_size], list[-labels:])
    
    return split_transactions

def user_profile_extract_equal_to_threshold(user_profile, threshold):
    """
    Extracts all rows from user_profile where the number of articles
    is equal to the given threshold.

    Parameters
    ----------
    user_profile : pd.DataFrame
        DataFrame with columns ['customer_id', 'article_id'], where 'article_id' is a list.
    threshold : int
        number of articles a customer must have to be kept.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only customers with article counts == threshold.
    """
    # Compute number of articles per customer
    user_profile['num_articles'] = user_profile['article_id'].apply(len)

    # Filter rows based on threshold
    preprocessed_user_profile = user_profile[user_profile['num_articles'] == threshold].copy()

    # Optionally drop helper column if not needed
    preprocessed_user_profile.drop(columns='num_articles', inplace=True)

    return preprocessed_user_profile

def remove_over_threshold(user_profile, threshold):
    """
    Removes all rows from user_profile where the number of articles
    exceed the given threshold.

    Parameters
    ----------
    user_profile : pd.DataFrame
        DataFrame with columns ['customer_id', 'article_id'], where 'article_id' is a list.
    threshold : int
        number of articles a customer must have to be kept.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only customers with article counts == threshold.
    """
    # Compute number of articles per customer
    user_profile['num_articles'] = user_profile['article_id'].apply(len)

    # Filter rows based on threshold
    preprocessed_user_profile = user_profile[user_profile['num_articles'] <= threshold].copy()

    # Optionally drop helper column if not needed
    preprocessed_user_profile.drop(columns='num_articles', inplace=True)

    return preprocessed_user_profile

def filter_transaction_list(min, quantile, transaction_list):
    # This function removes all users with less than 4 transactions and finds the maximum cut-off length for including 75%
    # of the data and removes all users that exceed that length.
    transaction_list = remove_under_threshold(transaction_list, min)
    cut_off = data_analyzer.get_cutoff_length_for_given_quantile(transaction_list, quantile)
    transaction_list = remove_over_threshold(transaction_list, cut_off)
    print(f"Users with less than {min} and more than {cut_off} transactions have been removed from the transaction_list dataset.")
    return transaction_list

def split_train_val_test_last_2(transaction_list_df):
    test = transaction_list_df.copy()
    val = transaction_list_df.copy()
    train = transaction_list_df.copy()

    val["article_id"] = val["article_id"].apply(lambda x: x[:-1])
    train["article_id"] = train["article_id"].apply(lambda x: x[:-2])

    return train, val, test

if __name__ == '__main__':
    transaction_list = pd.read_pickle("user_profiles.pkl")
    transaction_list = filter_transaction_list(4, 0.75, transaction_list)


