import os
import pickle
from transformer import prepare_dataset, train_model, get_all_unique_sids, is_model_trained
from transformers import BartTokenizer, BartForConditionalGeneration
import random

def take_subset_data(dict, frac=0.1, seed=42):
    if not dict:
        return {}
    rng = random.Random(seed)
    keys = list(dict.keys())
    k = max(1, int(len(keys) * frac))
    sampled_keys = rng.sample(keys, k)
    subset = {sample_k:dict[sample_k] for sample_k in sampled_keys}
    return subset

def train():
    """
    This trainer function is constructed to be run at the end with the enitire 
    train and val datasets.
    """

    print('Loading existing customer_transactions_train and customer_transactions_val')
    customer_transactions_train = {}
    customer_transactions_val = {}
    if os.path.exists('customer_transactions_train.pkl') and os.path.exists('customer_transactions_val.pkl'):
        with open('customer_transactions_train.pkl', 'rb') as f:
            customer_transactions_train = pickle.load(f)
        with open('customer_transactions_val.pkl', 'rb') as f:
            customer_transactions_val = pickle.load(f)

    print('The files are loaded!')

    customer_transactions_train_sub = take_subset_data(customer_transactions_train, frac=0.1, seed=42)
    customer_transactions_val_sub = take_subset_data(customer_transactions_val, frac=0.1, seed=42)


    print('Model is training ...') 
    window_size = 10
    unique_sids_train = get_all_unique_sids(customer_transactions_train_sub)
    unique_sids_val = get_all_unique_sids(customer_transactions_val_sub)
    all_unique_sids = set(unique_sids_train)
    for sid in unique_sids_val:
        all_unique_sids.add(sid)
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    tokenizer.add_tokens(list(all_unique_sids))

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    # resize model embeddings after adding token to tokenizer
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = prepare_dataset(customer_transactions_train_sub, window_size, tokenizer)
    val_dataset = prepare_dataset(customer_transactions_val_sub, window_size, tokenizer)
    train_model(train_dataset, model, val_dataset)

    model.save_pretrained('./bart-recommender/final_model')
    tokenizer.save_pretrained('./bart-recommender/final_model')

def main():
    train()

if __name__ == '__main__':
    main()