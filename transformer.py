import os
import numpy as np
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset,random_split

# setting device to cuda to utilize GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RecommendationDataset(Dataset):
    def __init__(self, sequences, tokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target = self.sequences[idx]
        source = self.tokenizer(input_seq, return_tensors="pt", padding="max_length", truncation=True, max_length=16)
        target = self.tokenizer(target, return_tensors="pt", padding="max_length", truncation=True, max_length=4)
        return{
            'input_ids': source['input_ids'].squeeze(),
            'attention_mask': source['attention_mask'].squeeze(),
            'labels': target['input_ids'].squeeze()
        }


def generate_random_user_histories(num_users=200, min_records=5, max_records=10, vector_size=3, min_value=1, max_value=999):
    """
    To generate randomly mockup user histories with SIDs:

        Output: A hashmap containing users history in form of:
            {"user1": [SID1, SID2, SID3, ...],
            "user2": ...}
            where each SID = [num1, num2, num3]
            and each num: min_value <= num1-3< max_value
    """
    user_histories = {}
    for i in range(1, num_users + 1):
        num_records = np.random.randint(min_records, max_records)
        history = [
            [np.random.randint(min_value, max_value) for _ in range(vector_size)]
            for _ in range(num_records)
        ]
        user_histories[f"user{i}"] = history
    return user_histories

# collect all unique SID strings to be added to the tokenizer's vocab
def get_all_unique_sids(user_histories):
    unique_sids =set()
    for history in user_histories.values():
        for vec in history:
            sid_str = ' '.join(map(str, vec))
            unique_sids.add(sid_str)
    return list(unique_sids)


# data augumentation
def create_sequences(history, window_size=3):
    sequences = []
    for i in range(len(history) - window_size):
        input_seq = ', '.join([' '. join(map(str, vec)) for vec in history[i:i+window_size]])
        target = ' '.join(map(str, history[i + window_size]))
        sequences.append((input_seq, target))
    
    return sequences

def prepare_dataset(user_histories, window_size, tokenizer):
    all_sequences = []
    for history in user_histories.values():
        all_sequences.extend(create_sequences(history, window_size))
    return RecommendationDataset(all_sequences, tokenizer)

# train the model
def train_model(train_dataset, model, eval_dataset=None, compute_metrics=None, eval_steps=100, patience=5):
    training_args = TrainingArguments(
        output_dir = './bart-recommender',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
        fp16=True if torch.cuda.is_available() else False, # to speed-up training when running on GPU
        eval_strategy = 'steps' if eval_dataset is not None else 'no',
        eval_steps= eval_steps if eval_dataset is not None else None,
        load_best_model_at_end = True if eval_dataset is not None else False,
        metric_for_best_model = 'eval_loss',
        greater_is_better = False
    )
    callbacks = []
    if eval_dataset is not None:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    trainer.train()

# inference (make prediction)
def recommended_next_sid(history, model, tokenizer, window_size=3, top_k=5):
    input_seq = ', '.join(history[-window_size:])
    inputs = tokenizer(input_seq, return_tensors="pt", truncation=True, max_length=16)
    device = next(model.parameters()).device  # Get model device
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tensors to model device
    output_ids = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=4,
        num_beams=max(top_k, 5),
        num_return_sequences=top_k,
        early_stopping=True
    )
    recommendations = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids] # return all tokens
    return recommendations

def is_model_trained(model_dir='./bart-recommender/final_model'):
    required_files = ['config.json', 'tokenizer_config.json']
    return all(os.path.isfile(os.path.join(model_dir, f)) for f in required_files)

def main():
    window_size=3

    if is_model_trained(): 
        print('The model is loaded...')
        model = BartForConditionalGeneration.from_pretrained('./bart-recommender/final_model')
        tokenizer = BartTokenizer.from_pretrained('./bart-recommender/final_model')
    else: 
        print('There is no pretrained model, the model will be trained ...')
        user_histories = generate_random_user_histories()

        unique_sids = get_all_unique_sids(user_histories)
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        tokenizer.add_tokens(unique_sids)

        model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        # resize model embeddings after adding token to tokenizer
        model.resize_token_embeddings(len(tokenizer))

        dataset = prepare_dataset(user_histories, window_size, tokenizer)

        # split dataset into train, val (80, 20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_model(train_dataset, model, val_dataset)

        model.save_pretrained('./bart-recommender/final_model')
        tokenizer.save_pretrained('./bart-recommender/final_model')

    # recommendation example
    test_history = ['110 450 228 503', '28 450 349 425']
    recommended_sid = recommended_next_sid(test_history, model, tokenizer, window_size)
    print('Recommended SID:', recommended_sid)

if __name__ == '__main__':
    main()
