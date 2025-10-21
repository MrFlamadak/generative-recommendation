import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def compute_item_user_matrix():
    transactions = pd.read_pickle("transactions_train.pkl")
    transactions["customer_id"] = transactions["customer_id"].astype(str)
    transactions["article_id"] = transactions["article_id"].astype(str)
    
    # map user ids and item ids to integer indices
    unique_users = transactions["customer_id"].unique()
    unique_items = transactions["article_id"].unique()
    user2idx = {u:i for i,u in enumerate(unique_users)}
    item2idx = {it:i for i,it in enumerate(unique_items)}

    transactions["u_idx"] = transactions["customer_id"].map(user2idx)
    transactions["i_idx"] = transactions["article_id"].map(item2idx)

    # build user-item interaction matrix
    num_users = len(unique_users)
    num_items = len(unique_items)
    rows = transactions["u_idx"].values
    cols = transactions["i_idx"].values
    data = np.ones(len(transactions), dtype=np.float32)

    interaction_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))

    # compute item-item similarity
    item_user_matrix = interaction_matrix.T
    item_similarity = cosine_similarity(item_user_matrix, dense_output=False)

    return unique_items, user2idx, interaction_matrix, item_similarity

def recommend_for_user(user_id, unique_items, user2idx, interaction_matrix, item_similarity, top_k=12):
   
    if user_id not in user2idx:
        return []
    
    u_idx = user2idx[user_id]
    user_vector = interaction_matrix[u_idx] # shape (1, num_items)

    bought_items_indices = user_vector.indices

    # add score to items similar to the bought items
    sim_scores = item_similarity[bought_items_indices].sum(axis=0) # shape (1, num_items)
    sim_scores = np.squeeze(np.array(sim_scores))

    # set the score for already bought to 0
    sim_scores[bought_items_indices] = 0

    # get top_k item indices
    top_k_indices = np.argpartition(-sim_scores, top_k)[:top_k]

    # map the top_k indices to corresponding items
    recommednded_items = [int(unique_items[i]) for i in top_k_indices]

    return recommednded_items

def main():
    unique_items, user2idx, interaction_matrix, item_similarity = compute_item_user_matrix()

    sample_users_id = pd.read_pickle('customers.pkl')["customer_id"][:10]
    for user_id in sample_users_id:
        recommended_items = recommend_for_user(user_id, unique_items, user2idx, interaction_matrix, item_similarity, top_k=12)
        print(f"User: {user_id}, recommended items: {recommended_items}")


if __name__ == "__main__":
    main()





