import numpy as np
#from minisom import MiniSom
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import plotly as plt
import pandas as pd
#import plotly.express as px


def print_data_stats(data):
    """
    Prints summary statistics of a dataset or embedding matrix.

    Parameters:
    -----------
    data : array-like (numpy array, list, or pandas DataFrame)
        Input data with shape (n_samples, n_features). Each row is a sample,
        each column is a feature.

    Prints:
    -------
    1. Shape of the data (n_samples, n_features)
    2. Per-feature mean and standard deviation
    3. Pairwise Euclidean distances and cosine similarities (mean, median, max)
       using a random subset of up to 500 samples for efficiency
    4. Number of principal components needed to explain 90% of variance
       (via PCA)
    """

    # Convert to numpy array if not already
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except Exception as e:
            raise ValueError(f"Cannot convert input data to numpy array: {e}")

    # 1. Basic shape and per-feature statistics
    rows, cols = data.shape
    num_features = min(cols, 5)
    print("Shape:", data.shape)
    print("Mean per feature (first 5 if >4 features):", np.round(data.mean(axis=0)[:num_features], 6))
    print("Std  per feature  (first 5 if >4 features):", np.round(data.std(axis=0)[:num_features], 6))

    # 2. Pairwise distances (using subset if large)
    n_samples = len(data)
    subset_idx = np.random.choice(n_samples, size=min(500, n_samples), replace=False)
    subset = data[subset_idx]

    # Euclidean distances
    euclid_dists = pairwise_distances(subset, metric="euclidean")
    # Cosine similarities
    cos_sims = 1 - pairwise_distances(subset, metric="cosine")
    # Use only upper triangle (unique pairs)
    iu = np.triu_indices_from(euclid_dists, k=1)

    print("Euclidean mean/min/max:",
          np.round(euclid_dists[iu].mean(), 6),
          np.round(euclid_dists[iu].min(), 6),
          np.round(euclid_dists[iu].max(), 6))
    print("Cosine similarity mean/median/max:",
          np.round(cos_sims[iu].mean(), 6),
          np.round(np.median(cos_sims[iu]), 6),
          np.round(cos_sims[iu].max(), 6))

    # 3. PCA to determine intrinsic dimensionality
    pca = PCA()
    pca.fit(data)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components_90 = np.argmax(cum_var >= 0.9) + 1

    print("Number of PCA components to cover 90% variance:", n_components_90)

    return
'''
def plot_som(vector_data, article_data):
    # Train MiniSom
    som = MiniSom(10, 10, vector_data.shape[1], sigma=0.5, learning_rate=0.5)

    som.train(vector_data, 1000)

    # Map each data point to its SOM position
    mapped = np.array([som.winner(x) for x in vector_data])

    # Create a dataframe for plotting
    plot_df = pd.DataFrame({
        "x": mapped[:, 0],
        "y": mapped[:, 1],
        "prod_name": article_data["prod_name"],
        "product_type_name": article_data["product_type_name"],
        "product_group_name": article_data["product_group_name"],
        "graphical_appearance_name": article_data["graphical_appearance_name"],
        "perceived_colour_value_name": article_data["perceived_colour_value_name"],
        "perceived_colour_master_name": article_data["perceived_colour_master_name"],
    })

    # --- Interactive scatter plot with hover info ---
    plot_df["x_jitter"] = plot_df["x"] + np.random.uniform(-0.01, 0.01, size=len(plot_df))
    plot_df["y_jitter"] = plot_df["y"] + np.random.uniform(-0.01, 0.01, size=len(plot_df))
    fig = px.scatter(
        plot_df,
        x="x_jitter",
        y="y_jitter",
        hover_data=[
            "prod_name",
            "product_type_name",
            "product_group_name",
            "graphical_appearance_name",
            "perceived_colour_value_name",
            "perceived_colour_master_name",
        ],
        title="MiniSom Product Map (hover to inspect items)",
        color="product_type_name",  # Optional: color by a meaningful feature
    )

    fig.update_traces(marker=dict(size=10, opacity=0.7))
    fig.update_layout(width=800, height=700)
    fig.show()

    #print(sem_data.shape)
    print(np.unique([som.winner(x) for x in vector_data], axis=0).shape)
'''

def print_article_dataset_infostats():
    article_df = pd.read_pickle("data/articles.pkl")

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

def print_transaction_list_dataset_infostats():
    user_profile_train_df = pd.read_pickle("data/transaction_list_train.pkl")
    # training, test = dh.create_test_and_training_user_profiles(0.8)

    # Compute lengths of article_id arraystr

    user_profile_train_df['num_articles'] = user_profile_train_df['article_id'].apply(len)

    # Count how many customers have each article count
    counts = user_profile_train_df['num_articles'].value_counts().sort_index()

    #Find cut-off for 75%-80% of data

    # [5, 3, 9 ...]
    print(counts.head())
    quantile_75 = 0.75
    quantile_80 = 0.80
    quantile_85 = 0.85
    quantile_90 = 0.90
    total = len(user_profile_train_df)

    sum = 0
    num_of_articles = 2
    for i in range(0, len(counts)):
        article_count = counts.iloc[i]
        sum = sum + article_count
        current_fraction = sum/total
        if(current_fraction >= quantile_75):
            print(f"{round(current_fraction, 3)*100}% of users have {num_of_articles} or fewer transactions")
            break
        num_of_articles = num_of_articles + 1

    return
def get_cutoff_length_for_given_quantile(transaction_list_df, quantile=0.75):
    transaction_list_df['num_articles'] = transaction_list_df['article_id'].apply(len)

    # Count how many customers have each article count
    counts = transaction_list_df['num_articles'].value_counts().sort_index()
    cutoff_length = 0
    num_of_articles = counts.index[0]
    sum = 0
    total = len(transaction_list_df)

    for i in range(0, len(counts)):
        article_count = counts.iloc[i]
        sum = sum + article_count
        current_fraction = sum/total
        if(current_fraction >= quantile):
            print(f"{round(current_fraction, 3)*100}% of users have {num_of_articles} or fewer transactions")
            return num_of_articles
        num_of_articles = num_of_articles + 1

    return num_of_articles


if __name__ == '__main__':
    '''embeddings_data = np.load("SBERT_embeddings_fulldata.npy")
    print("EMBEDDINGS DATA STATS")
    print_data_stats(embeddings_data)
    sem_data, article_data = dh.get_random_item_to_sem_ids(50)
    print("\nSEMANTIC ID DATA STATS")
    print_data_stats(sem_data)
    plot_som(sem_data, article_data)'''
    #print_transaction_list_dataset_infostats()
    #list = pd.read_pickle("user_profiles.pkl")
    #cut_off = get_cutoff_length_for_given_quantile(list, 0.75)
    print_article_dataset_infostats()


