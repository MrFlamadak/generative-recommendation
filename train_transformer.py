import os
import pickle
from transformer import prepare_dataset, train_model, get_all_unique_sids, is_model_trained
from transformers import BartTokenizer, BartForConditionalGeneration

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



    print('Model is training ...')
    window_size = 3
    unique_sids_train = get_all_unique_sids(customer_transactions_train)
    unique_sids_val = get_all_unique_sids(customer_transactions_val)
    all_unique_sids = set(unique_sids_train)
    for sid in unique_sids_val:
        all_unique_sids.add(sid)
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    tokenizer.add_tokens(list(all_unique_sids))

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    # resize model embeddings after adding token to tokenizer
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = prepare_dataset(customer_transactions_train, window_size, tokenizer)
    val_dataset = prepare_dataset(customer_transactions_val, window_size, tokenizer)
    train_model(train_dataset, model, val_dataset)

    model.save_pretrained('./bart-recommender/final_model')
    tokenizer.save_pretrained('./bart-recommender/final_model')

def main():
    train()

if __name__ == '__main__':
    main()