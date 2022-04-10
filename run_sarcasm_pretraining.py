import torch
import argparse
import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForPreTraining


class SarcasmDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset = self.dataset.sample(
            frac=1, random_state=args.seed).reset_index(drop=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.transform(self.dataset.iloc[index])

    def transform(self, row):
        return row['text'], row['is_sarcasm']


def parse_arguments():
    parser = argparse.ArgumentParser(description="Sarcasm pretraining script")
    parser.add_argument("--input_files",
                        default=None,
                        required=True,
                        nargs='+',
                        help="The input files. Should contain csv files for the task.")

    parser.add_argument("--bert_model",
                        default="bert-base-uncased",
                        type=str,
                        help="Bert model type or path to weights directory.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization.")

    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs",
                        default=5,
                        type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--tokenizer_path",
                        default='bert-base-uncased',
                        type=str,
                        help="The path to the tokenizer.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=8,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")

    args = parser.parse_args()
    return args


def get_datasets():
    ds = pd.DataFrame()
    for input_file in args.input_files:
        ds = ds.append(pd.read_csv(input_file))
    ds_train, ds_test = train_test_split(
        ds, test_size=0.2, random_state=args.seed)
    ds_train, ds_eval = train_test_split(
        ds_train, test_size=0.25, random_state=args.seed)

    ds_train = ds_train.reset_index(drop=True)
    ds_test = ds_test.reset_index(drop=True)
    ds_eval = ds_eval.reset_index(drop=True)
    return ds_train, ds_eval, ds_test


def mask_inputs(tokenized_ids):
    tags = [101, 102, 103, 100, 0]

    for row_id in range(len(tokenized_ids)):
        for col_id in range(len(tokenized_ids[row_id])):
            if tokenized_ids[row_id][col_id] not in tags:
                rand = torch.rand(1)
                if rand <= 0.15:
                    tokenized_ids[row_id][col_id] = 103

    return tokenized_ids


def train(model, tokenizer, ds_train, ds_eval):
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    grad_batches = (len(ds_train) // args.gradient_accumulation_steps) * \
        args.gradient_accumulation_steps
    loss_history = {
        'train': [],
        'eval': []
    }
    min_eval_loss = float('inf')
    for epoch in range(args.num_train_epochs):
        avg_loss = 0
        model.train()
        optim.zero_grad()
        with tqdm(ds_train, unit='batch', leave=True, position=0) as tepoch:
            for batch_idx, (text, label) in enumerate(tepoch):
                inputs = tokenizer(
                    text, padding=True, truncation=True, max_length=512, return_tensors='pt')
                lm_labels = inputs['input_ids'].to(device)
                sarcasm_labels = label.to(device)
                masked_tokens = mask_inputs(inputs['input_ids']).to(device)
                attention_mask = inputs['attention_mask'].to(device)

                outputs = model(masked_tokens, attention_mask=attention_mask,
                                labels=lm_labels, next_sentence_label=sarcasm_labels)
                loss = outputs.loss
                avg_loss += (loss.item() / len(ds_train))

                # Scale loss for gradient accumulation steps
                if batch_idx >= grad_batches:
                    loss = loss / (len(ds_train) - grad_batches)
                else:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                if ((batch_idx + 1) % args.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(ds_train)):
                    optim.step()
                    optim.zero_grad()

                tepoch.set_description(f'Epoch {epoch}')
                tepoch.set_postfix(loss=loss.item())
                torch.cuda.empty_cache()

        loss_history['train'].append(avg_loss)
        avg_loss = 0
        model.eval()
        with torch.no_grad():
            with tqdm(ds_eval, unit='batch', leave=True, position=0) as eepoch:
                for batch_idx, (text, label) in enumerate(eepoch):
                    inputs = tokenizer(
                        text, padding=True, truncation=True, max_length=512, return_tensors='pt')
                    lm_labels = inputs['input_ids'].to(device)
                    sarcasm_labels = label.to(device)
                    masked_tokens = mask_inputs(inputs['input_ids']).to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    outputs = model(masked_tokens, attention_mask=attention_mask,
                                    labels=lm_labels, next_sentence_label=sarcasm_labels)
                    loss = outputs.loss
                    avg_loss += (loss.item() / len(ds_eval))

                    eepoch.set_description(f'Eval Epoch {epoch}')
                    eepoch.set_postfix(loss=loss.item())
                    torch.cuda.empty_cache()

        loss_history['eval'].append(avg_loss)
        if avg_loss < min_eval_loss:
            min_eval_loss = avg_loss
            model.save_pretrained(args.output_dir)

    return loss_history


def main():
    print("running with input files: \n", args.input_files)
    print("Setting up random seeds ...")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print("Setting up datasets ...")
    ds_train, ds_eval, ds_test = get_datasets()
    ds_train.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
    ds_test.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)
    ds_eval.to_csv(os.path.join(args.output_dir, 'eval.csv'), index=False)
    ds_train = SarcasmDataset(ds_train)
    ds_eval = SarcasmDataset(ds_eval)
    print("Setting up dataloaders ...")
    train_loader = DataLoader(
        ds_train, batch_size=args.train_batch_size, shuffle=True)
    eval_loader = DataLoader(
        ds_eval,  batch_size=args.train_batch_size, shuffle=True)
    print("Setting up tokenizer ...")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    print("Setting up model ...")
    model = BertForPreTraining.from_pretrained(args.bert_model)
    print("Starting training...")
    loss_history = train(model, tokenizer, train_loader, eval_loader)
    print("Saving loss history...")
    loss_history = pd.DataFrame(
        data=loss_history, index=range(args.num_train_epochs))
    loss_history.to_csv(os.path.join(
        args.output_dir, 'loss_history.csv'), index=False)


if __name__ == "__main__":
    args = parse_arguments()
    main()
