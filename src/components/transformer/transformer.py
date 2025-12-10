import os
import pickle
import json, csv
import numpy as np
import torch
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)
from torch.utils.data import Dataset

# device for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Q = 6  # number of codebooks, length of SID


class RecommendationDataset(Dataset):
    """
    sequences: list of tuples (input_seq_str, target_seq_str)
               where input_seq_str is a space-separated list of item-tokens
               (already right-padded to fixed item count when prepared),
               and target_seq_str is a single item-token string.
    tokenizer: HF tokenizer
    input_length_items: number of item-tokens expected in encoder input (max_length)
    """

    def __init__(self, sequences, tokenizer, input_length_items):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.input_length_items = input_length_items

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target = self.sequences[idx]
        # Tokenize encoder input (right padding is handled by tokenizer.padding_side)
        enc = self.tokenizer(
            input_seq,
            padding="max_length",
            truncation="longest_first",
            max_length=self.input_length_items * Q,  #
            return_tensors="pt"
        )

        # Tokenize decoder target (single token expected)
        dec = self.tokenizer(
            target,
            padding="max_length",
            truncation=False,
            max_length=Q + 2,  # single SID, which contains 6 items + EOS, BOS
            return_tensors="pt"
        )

        labels = dec["input_ids"].squeeze(0).clone()  # shape (1,) or (max_len,)
        # replace pad_token_id with -100 for loss ignore
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        return {
            "input_ids": enc["input_ids"].squeeze(0),  # tensor shape (seq_len,)
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels  # tensor dtype long
        }


#  Synthetic data generation
def generate_random_user_histories(num_users=10, min_records=4, max_records=37, vector_size=6, min_value=1,
                                   max_value=999):
    user_histories = {}
    for i in range(1, num_users + 1):
        num_records = np.random.randint(min_records, max_records + 1)  # inclusive upper bound
        history = [
            [np.random.randint(min_value, max_value) for _ in range(vector_size)]
            for _ in range(num_records)
        ]
        user_histories[f"user{i}"] = history
    return user_histories


def sid_string_from_vec(vec):
    """Convert numeric vector to string vector "C0_12 C1_13 C2_14 C3_15 C4_16 C5_0"""
    return " ".join([f"C{i}_{v}" for i, v in enumerate(vec)])


def sid_vec_from_string(decoded_string):
    """Convert string vector "C0_12 C1_13 C2_14 C3_15 C4_16 C5_0" vector to a numeric vector"""
    tokens = decoded_string.split()
    vec = []
    for tok in tokens:
        if tok.startswith("C") and "_" in tok:
            try:
                q, val = tok.split("_")
                vec.append(int(val))
            except:
                pass
    return vec[:Q]  # to make sure about correct length


def get_all_unique_sid(user_histories):
    """
    This function is used for the test case with mock-up dada
    """
    unique_sids = set()
    for history in user_histories.values():
        for vec in history:
            unique_sids.add(sid_string_from_vec(vec))
    return list(unique_sids)


def get_all_unique_tokens_in_sids(item_to_semantics):
    """
    This function gets the item_to_semantics and returns a set of
    distinct tokens from all SIDs
    """
    unique_tokens_in_sids = set()
    for sid in item_to_semantics.values():  # 12 13 14 15
        # unique_sids.add(sid_string_from_vec(sid)) # C0_1 ..
        for i, element in enumerate(sid):
            # print("YOYO:", element)
            unique_tokens_in_sids.add(f"C{i}_{element}")

    return list(unique_tokens_in_sids)


# dataset creation
def prepare_dataset(user_histories, window_size, tokenizer):
    """
    Build (input, target) pairs.
    - window_size: the window includes the target, so encoder input uses (window_size - 1) items.
    - The encoder input will be right-padded to exactly input_len items (item tokens).
    """
    input_len = max(1, window_size - 1)  # number of item tokens to use as encoder input
    pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else "<PAD_ITEM>"

    all_sequences = []
    for history in user_histories.values():
        if len(history) < 2:
            continue

        # target is last item in this simplified setup
        label_vec = history[-1]
        target_token = sid_string_from_vec(label_vec)

        # take up to input_len items before the last one
        input_items_vecs = history[:-1][-input_len:]
        item_tokens = [sid_string_from_vec(v) for v in input_items_vecs]
        pad_needed = input_len - len(item_tokens)
        # padded_items = item_tokens + [pad_token] * Q * pad_needed # right padded
        padded_items = []
        padded_items.extend(item_tokens)
        for _ in range(pad_needed):
            padded_items.extend([pad_token] * Q)  # join tokens with spaces (this yields exactly input_len tokens)

        input_seq = " ".join(padded_items)
        all_sequences.append((input_seq, target_token))

    return RecommendationDataset(all_sequences, tokenizer, input_len)


# train
def train_model(train_dataset, model, eval_dataset=None, eval_steps=200, patience=5, grad_accum_steps=1, num_workers=4):
    training_args = TrainingArguments(
        output_dir='./../models/bart-recommender_iteration2',
        num_train_epochs=8,  # increase to 7 or 8 to see the trend
        per_device_train_batch_size=512,  # increase if needed
        gradient_accumulation_steps=grad_accum_steps,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=200,
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
        fp16=True if torch.cuda.is_available() else False,  # mixed precision on CUDA
        optim='adamw_torch',
        eval_strategy='steps' if eval_dataset is not None else 'no',
        eval_steps=eval_steps if eval_dataset is not None else None,
        load_best_model_at_end=True if eval_dataset is not None else False,
        metric_for_best_model='eval_loss',
        greater_is_better=False
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)] if eval_dataset is not None else []

    # enable gradient checkpointing if supported
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    # Data collator ensures batch padding and proper label padding
    data_collator = DataCollatorForSeq2Seq(tokenizer=train_dataset.tokenizer, label_pad_token_id=-100)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        data_collator=data_collator
    )

    trainer.train()

    # log trainer.state.log_history
    log_history = trainer.state.log_history
    os.makedirs('./../models/bart-recommender_iteration2', exist_ok=True)

    # dump full log history (JSON)
    with open('./../models/bart-recommender_iteration2/training_log_history.json', 'w') as fh:
        json.dump(log_history, fh, indent=2)

    # extract per-step train losses and eval_losses
    train_losses = [{'step': rec.get('step'), 'loss': rec['loss']} for rec in log_history if 'loss' in rec]
    eval_losses = [{'step': rec.get('step'), 'eval_loss': rec['eval_loss']} for rec in log_history if
                   'eval_loss' in rec]

    # save CSVs for easy plotting
    with open('./../models/bart-recommender_iteration2/train_losses.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['step', 'loss'])
        writer.writeheader()
        writer.writerows(train_losses)

    with open('./../models/bart-recommender_iteration2/eval_losses.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['step', 'eval_loss'])
        writer.writeheader()
        writer.writerows(eval_losses)


# inference
def recommended_next_sid(history, model, tokenizer, window_size=36, top_k=1):
    input_len = max(1, window_size - 1)
    pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else "<PAD_ITEM>"

    # take up to input_len items before the last one
    input_items = history[-input_len:]
    item_tokens = []
    for it in input_items:
        if isinstance(it, str):
            s = it.strip()
            if not s:
                continue
            parts = [int(x) for x in s.split() if x.strip()]
            item_tokens.append(sid_string_from_vec(parts))
        else:
            continue

    pad_needed = input_len - len(item_tokens)

    # build input tokens
    tokens = []
    for item in item_tokens:
        tokens.extend(item.split())
    for _ in range(pad_needed):
        tokens.extend([pad_token] * Q)

    input_seq = " ".join(tokens)

    enc = tokenizer(input_seq, return_tensors="pt",
                    max_length=input_len * Q,
                    padding="max_length",
                    truncation="longest_first")

    enc = {k: v.to(device) for k, v in enc.items()}

    outputs = model.generate(
        enc["input_ids"],
        attention_mask=enc["attention_mask"],
        num_beams=max(top_k, 1),
        num_return_sequences=top_k,
        # max_length=Q + 8,
        max_new_tokens=Q + 8,
        early_stopping=True
    )

    results = []
    for seq in outputs:
        decoded = tokenizer.decode(seq, skip_special_tokens=True)
        vec = sid_vec_from_string(decoded)
        results.append(vec)

    return results


def is_model_trained(model_dir='./../models/bart-recommender_iteration2/final_model'):
    print(os.getcwd())
    required_files = ["config.json", "tokenizer_config.json"]
    return all(os.path.isfile(os.path.join(model_dir, f)) for f in required_files)


def main():
    window_size = 36  # max window (includes target); encoder gets window_size-1 items

    if is_model_trained():
        print("Loading pretrained model...")
        model = BartForConditionalGeneration.from_pretrained('./../models/bart-recommender_iteration2/final_model')
        tokenizer = BartTokenizer.from_pretrained("'./../models/bart-recommender_iteration2/final_model'")
    else:
        print("Training new model (synthetic data)...")
        # generate synthetic histories for train/val
        user_histories_train = generate_random_user_histories(num_users=10)
        user_histories_val = generate_random_user_histories(num_users=1)

        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

        # ensure pad token exists (single-token)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<PAD_ITEM>"})

        tokenizer.padding_side = "right"

        # collect all SIDs and add them (single tokens)
        sids_train = get_all_unique_sid(user_histories_train)
        sids_val = get_all_unique_sid(user_histories_val)
        all_sids = list(set(sids_train) | set(sids_val))

        tokenizer.add_tokens(all_sids)

        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        # resize embeddings after we changed tokenizer
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

        # prepare datasets
        train_dataset = prepare_dataset(user_histories_train, 35, tokenizer)
        val_dataset = prepare_dataset(user_histories_val, 36, tokenizer)
        # for i, (input, target) in enumerate(val_dataset.sequences):
        #     print(f"{i}: INPUT = {input} --> TARGET = {target}\n")
        #     print(len(input),", " ,len(target))
        # return

        # train
        train_model(train_dataset, model, eval_dataset=val_dataset, eval_steps=4, patience=3)

        # save
        os.makedirs('./../models/bart-recommender_iteration2/final_model', exist_ok=True)
        model.save_pretrained('./../models/bart-recommender_iteration2/final_model')
        tokenizer.save_pretrained('./../models/bart-recommender_iteration2/final_model')

    # Example inference (use SID tokens in history format)
    # Create an example history of SID tokens (must match SID token string format)
    test_history = ['110 450 228 503', '28 450 349 425', '28 450 349 425']
    model = model.to(device)
    recs = recommended_next_sid(test_history, model, tokenizer, window_size=window_size, top_k=5)
    print("Recommended SIDs:", recs)


if __name__ == "__main__":
    main()
