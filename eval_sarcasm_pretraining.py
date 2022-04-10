import argparse
import random
import torch
import numpy as np
import pandas as pd
import subprocess
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForPreTraining
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
    parser = argparse.ArgumentParser(
        description="Evaluation of sarcasm pretraining")
    parser.add_argument('--model_path', type=str,
                        required=True,
                        help="Path to the trained model.")

    parser.add_argument("--input_files", default=None,
                        required=True, nargs='+',
                        help="The input files. Should contain csv files for the task.")

    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for the DataLoader.")

    parser.add_argument('--output_path', type=str,
                        required=True, help="Output path to log metrics.")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization.")

    parser.add_argument("--tokenizer_path",
                        default='bert-base-uncased', type=str,
                        help="The path to the tokenizer.")

    args = parser.parse_args()
    return args


def get_datasets():
    ds = pd.DataFrame()
    for input_file in args.input_files:
        ds = ds.append(pd.read_csv(input_file))
    return ds


def evaluate(model, dataset, tokenizer):
    model.eval()
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    y_true = []
    y_pred = []
    with torch.no_grad():
        with tqdm(dataset, unit='batch', leave=True, position=0) as tepoch:
            for batch_idx, (text, label) in enumerate(tepoch):
                inputs = tokenizer(
                    text, padding=True, truncation=True, max_length=512, return_tensors='pt')
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                y_true = y_true + list(label.numpy())
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                y_pred = y_pred + \
                    list(
                        outputs['seq_relationship_logits'].detach().cpu().max(1).indices.numpy())
                torch.cuda.empty_cache()

    acc_score = accuracy_score(y_true, y_pred)
    prec_score = precision_score(y_true, y_pred)
    rec_score = recall_score(y_true, y_pred)
    f_score = f1_score(y_true, y_pred)
    print("Printing metrics...")
    print("Accuracy: ", acc_score)
    print("Precision: ", prec_score)
    print("Recall: ", rec_score)
    print("F1: ", f_score)
    print("Saving metrics...")

    metrics = pd.DataFrame(
        {'accuracy': [acc_score], 'precision': [prec_score], 'recall': [rec_score], 'f1': [f_score]})
    metrics.to_csv(args.output_path, index=False)


def main():

    print("running with input files: \n", args.input_files)
    # Setting up random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Setting up datasets
    test_dataset = get_datasets()
    test_dataset = SarcasmDataset(test_dataset)
    print("Datsets configured...")

    # Setting up dataloaders
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)
    print("DataLoader configured...")

    # Setting up tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)

    # Setting up the model
    model = BertForPreTraining.from_pretrained(args.model_path)
    print("Model configured...")

    print("Starting Evaluation...")
    evaluate(model, test_loader, tokenizer)


if __name__ == "__main__":
    args = parse_arguments()
    main()
