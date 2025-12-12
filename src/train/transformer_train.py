import os
import pickle
from src.components.transformer import prepare_dataset, train_model, get_all_unique_tokens_in_sids
from transformers import BartTokenizer, BartForConditionalGeneration
import random


def take_subset_data(dict, frac=0.1, seed=42):
    if not dict:
        return {}
    rng = random.Random(seed)
    keys = list(dict.keys())
    k = max(1, int(len(keys) * frac))
    sampled_keys = rng.sample(keys, k)
    subset = {sample_k: dict[sample_k] for sample_k in sampled_keys}
    return subset


def start_training():
    """
    This trainer function is constructed to be run at the end with the enitire
    train and val datasets.
    """

    print('Loading existing customer_transactions_train and customer_transactions_val')
    customer_transactions_train = {}
    customer_transactions_val = {}
    if os.path.exists('./../data/customer_transactions_train.pkl') and os.path.exists('./../data/customer_transactions_val.pkl'):
        with open('./../data/customer_transactions_train.pkl', 'rb') as f:
            customer_transactions_train = pickle.load(f)
        with open('./../data/customer_transactions_val.pkl', 'rb') as f:
            customer_transactions_val = pickle.load(f)

    print('The files are loaded!')
    # print(f"Length of customer_transactions_train: {[len(customer_transactions_train[key]) for key in list(customer_transactions_train.keys())[:10]]}")
    # print(f"Length of customer_transactions_val: {[len(customer_transactions_val[key]) for key in list(customer_transactions_val.keys())[:10]]}")
    # customer_transactions_train_sub = take_subset_data(customer_transactions_train, frac=0.5, seed=42)
    customer_transactions_val_sub = take_subset_data(customer_transactions_val, frac=0.2, seed=42)

    # print(f"Length of customer_transactions_train_sub: {len(customer_transactions_train_sub.keys())}")
    # print(f"Length of customer_transactions_val_sub: {len(customer_transactions_val_sub.keys())}")

    print('Model is training ...')

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    # ensure pad token exists (single-token)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD_ITEM>"})

    tokenizer.padding_side = "right"

    item_to_semantics = {}
    if os.path.exists(
            './../data/semantic_ids/item_2_semantic.pkl'):
        with open(
                './../data/semantic_ids/item_2_semantic.pkl',
                'rb') as f:
            item_to_semantics = pickle.load(f)
    all_sids = get_all_unique_tokens_in_sids(item_to_semantics)

    # add tokens
    tokenizer.add_tokens(all_sids)

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    # resize embeddings after we changed tokenizer
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = prepare_dataset(customer_transactions_train, window_size=35, tokenizer=tokenizer)
    val_dataset = prepare_dataset(customer_transactions_val_sub, window_size=36, tokenizer=tokenizer)
    # for i, (input, target) in enumerate(train_dataset.sequences):
    #     print(f"{i}: INPUT = {input} --> TARGET = {target}\n")
    #     print(len(input.split()),", " ,len(target.split()))
    #     break
    # for i, (input, target) in enumerate(val_dataset.sequences):
    #     print(f"{i}: INPUT = {input} --> TARGET = {target}\n")
    #     print(len(input.split()),", " ,len(target.split()))
    #     break
    # return
    train_model(train_dataset, model, val_dataset)

    # save
    os.makedirs("./../models/bart-recommender_iteration2/final_model", exist_ok=True)
    model.save_pretrained('./../models/bart-recommender_iteration2/final_model')
    tokenizer.save_pretrained('./../models/bart-recommender_iteration2/final_model')


def main():
    start_training()


if __name__ == '__main__':
    main()