import argparse
import random
import torch
import os
import numpy as np
import pandas as pd
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import RobertaConfig, RobertaTokenizer, get_linear_schedule_with_warmup, RobertaModel
from modelling.mixout import MixLinear


class ImplicitHateDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset = self.dataset.sample(
            frac=1, random_state=args.seed).reset_index(drop=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.transform(self.dataset.iloc[index])

    def transform(self, row):
        if row['class'] == 'not_hate':
            return row['post'], 0
        elif row['class'] == 'implicit_hate':
            return row['post'], 1


class IHDModel(torch.nn.Module):
    def __init__(self):
        super(IHDModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(args.roberta_model_path)
        self.fc1 = torch.nn.Linear(768, 100)
        self.act1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(100, 100)
        self.act2 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(100, args.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask)
        x = outputs[0]
        x = self.fc1(x[:, 0, :])
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x


def parse_arguments():
    parser = argparse.ArgumentParser(description="Stage 1 training script")
    parser.add_argument("--input_files",
                        default=None,
                        required=True,
                        nargs='+',
                        help="The input files. Should contain csv files for the task.")

    parser.add_argument("--roberta_model_path",
                        default="roberta-base",
                        type=str,
                        help="Roberta model type or path to weights directory.")

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
                        default=8,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--tokenizer_path",
                        default='roberta-base',
                        type=str,
                        help="The path to the tokenizer.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument("--num_labels",
                        type=int,
                        default=2,
                        help="Number of labels.")

    parser.add_argument("--reinit_n_layers",
                        type=int,
                        default=5,
                        help="Re-init last n encoder layers")

    parser.add_argument("--mixout_rate",
                        default=0.7,
                        type=float,
                        help="Mixout regularization rate")

    parser.add_argument("--llrd_type",
                        default='s',
                        choices=['s', 'i', 'a'],
                        help="LLRD type")

    args = parser.parse_args()
    return args


def get_optimizer_params(model, type='s'):
    # differential learning rate and weight decay
    param_optimizer = list(model.named_parameters())
    learning_rate = args.learning_rate
    no_decay = ['bias']
    if type == 's':
        optimizer_parameters = filter(
            lambda x: x.requires_grad, model.parameters())
    elif type == 'i':
        optimizer_parameters = [
            {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "roberta" not in n],
             'lr': learning_rate*10,
             'weight_decay':0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and "roberta" not in n],
             'lr': learning_rate*10,
             'weight_decay':0.0},
        ]
    elif type == 'a':
        group1 = ['embeddings.', 'layer.0.',
                  'layer.1.', 'layer.2.', 'layer.3.']
        group2 = ['layer.4.', 'layer.5.', 'layer.6.', 'layer.7.']
        group3 = ['layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
        optimizer_parameters = [
            {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(
                nd in n for nd in group1)], 'weight_decay': 0.01, 'lr': learning_rate/2.6},
            {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(
                nd in n for nd in group2)], 'weight_decay': 0.01, 'lr': learning_rate},
            {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(
                nd in n for nd in group3)], 'weight_decay': 0.01, 'lr': learning_rate*2.6},
            {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(
                nd in n for nd in group1)], 'weight_decay': 0.0, 'lr': learning_rate/2.6},
            {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(
                nd in n for nd in group2)], 'weight_decay': 0.0, 'lr': learning_rate},
            {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(
                nd in n for nd in group3)], 'weight_decay': 0.0, 'lr': learning_rate*2.6},
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay) and "roberta" not in n], 'lr':learning_rate*10, 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay) and "roberta" not in n], 'lr':learning_rate*10, 'weight_decay': 0.0},
        ]
    return optimizer_parameters


def initialize_mixout(model):
    no_mixout = ['fc1', 'fc2', 'fc3', 'query', 'key', 'value']
    if args.mixout_rate > 0:
        print('Initializing Mixout Regularization')
        for sup_module in model.modules():
            for name, module in sup_module.named_children():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
                if isinstance(module, nn.Linear) and not(name in no_mixout):
                    target_state_dict = module.state_dict()
                    bias = True if module.bias is not None else False
                    new_module = MixLinear(
                        module.in_features, module.out_features, bias, target_state_dict[
                            "weight"], args.mixout_rate
                    )
                    new_module.load_state_dict(target_state_dict)
                    setattr(sup_module, name, new_module)


def re_init_layers(model, config):
    if args.reinit_n_layers > 0:
        print(f'Reinitializing Last {args.reinit_n_layers} Layers ...')
        encoder_temp = getattr(model, 'roberta')
        for layer in encoder_temp.encoder.layer[-args.reinit_n_layers:]:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(
                        mean=0.0, std=config.initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.normal_(
                        mean=0.0, std=config.initializer_range)
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)


def get_datasets():
    ds = pd.DataFrame()
    for input_file in args.input_files:
        ds = ds.append(pd.read_csv(input_file, sep='\t'))

    ds = ds.loc[ds['class'] != 'explicit_hate']

    ds_train, ds_test = train_test_split(
        ds, test_size=0.2, random_state=42)
    ds_train, ds_eval = train_test_split(
        ds_train, test_size=0.25, random_state=42)

    ds_train = ds_train.reset_index(drop=True)
    ds_test = ds_test.reset_index(drop=True)
    ds_eval = ds_eval.reset_index(drop=True)
    return ds_train, ds_eval, ds_test


def calculate_label_weights(ds):
    label_counts = ds['class'].value_counts()
    label_weights = {}
    for label in label_counts.index:
        label_weights[label] = len(ds) / label_counts[label]

    label_weights = [label_weights['not_hate'], label_weights['implicit_hate']]
    return label_weights


def train(model, tokenizer, ds_train, ds_eval):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    loss_history = {
        'train': [],
        'eval': []
    }
    min_eval_loss = float('inf')
    params = get_optimizer_params(model, 'a')
    optim = torch.optim.AdamW(params, args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=len(ds_train), num_training_steps=len(ds_train) * args.num_train_epochs)
    for epoch in range(args.num_train_epochs):
        avg_loss = 0
        optim.zero_grad()
        with tqdm(ds_train, unit='batch', leave=True, position=0) as tepoch:
            for batch_idx, (text, label) in enumerate(tepoch):
                with torch.no_grad():
                    inputs = tokenizer(
                        text, padding=True, truncation=True, max_length=512, return_tensors='pt')
                    input_ids = inputs['input_ids'].to(device)
                    stg1_labels = label.to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, stg1_labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
                scheduler.step()
                avg_loss += (loss.item() / len(ds_train))
                tepoch.set_description(f'Train Epoch {epoch}')
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
                    input_ids = inputs['input_ids'].to(device)
                    stg1_labels = label.to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask)
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(outputs, stg1_labels)
                    avg_loss += (loss.item() / len(ds_eval))
                    eepoch.set_description(f'Eval Epoch {epoch}')
                    eepoch.set_postfix(loss=loss.item())
                    torch.cuda.empty_cache()

        loss_history['eval'].append(avg_loss)
        if avg_loss < min_eval_loss:
            min_eval_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(
                args.output_dir, 'model.pt'))

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
    ds_train = ImplicitHateDataset(ds_train)
    ds_eval = ImplicitHateDataset(ds_eval)
    print("Setting up dataloaders ...")
    ds_train = DataLoader(
        ds_train, batch_size=args.train_batch_size, shuffle=True)
    ds_eval = DataLoader(
        ds_eval,  batch_size=args.train_batch_size, shuffle=True)
    print("Setting up tokenizer ...")
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_path)
    print("Setting up model ...")
    config = RobertaConfig.from_pretrained(args.roberta_model_path)
    model = IHDModel()
    re_init_layers(model, config)
    initialize_mixout(model)
    print("Starting training...")
    loss_history = train(model, tokenizer, ds_train, ds_eval)
    print("Saving loss history...")
    loss_history = pd.DataFrame(
        data=loss_history, index=range(args.num_train_epochs))
    loss_history.to_csv(os.path.join(
        args.output_dir, 'loss_history.csv'), index=False)


if __name__ == "__main__":
    args = parse_arguments()
    main()
