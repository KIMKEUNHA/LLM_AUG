import pytorch_lightning as pl
import torch
import datasets
from datasets import load_dataset,concatenate_datasets

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import json
import pandas as pd
import numpy as np
class IMDB(Dataset):
    def __init__(self, tok, text, label):
        self.tok = tok
        self.text = text
        self.label = label

        assert len(self.text) == len(self.label)
        print(f"Load {len(self.label)} data.")

    def __getitem__(self, idx):
        src = self.tok(
            self.text[idx], truncation=True, padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": src["input_ids"].squeeze(),
            "token_type_ids": src["token_type_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "labels": torch.tensor(self.label[idx]),
        }

    def __len__(self):
        return len(self.label)


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer

    def prepare_data(self):
        """Only called from the main process for downloading dataset"""
        load_dataset(self.cfg.dataset_name, split="train")
        load_dataset(self.cfg.dataset_name, split="test")

    def setup(self, stage: str):
        
        label_mapping = {'negative':0,'positive':1}
        if stage == "fit":
            if self.cfg.dataset == "origin":        
                dset = load_dataset(self.cfg.dataset_name, split="train")
                self.trn_dset = IMDB(self.tokenizer, dset["text"], dset["label"])
                dset = load_dataset(self.cfg.dataset_name, split="test")
                self.val_dset = IMDB(self.tokenizer, dset["text"], dset["label"])
            elif self.cfg.dataset == "aug":     
                dset = load_dataset('json', data_files="data/IMDB.jsonl")
                df = pd.DataFrame(dset['train'])
                df['label'] = df['label'].map(label_mapping)
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                df['label'] = df['label'].astype(int)
                dset =  datasets.Dataset.from_pandas(df)
                self.trn_dset = IMDB(self.tokenizer, dset["text"], dset["label"])
                dset = load_dataset(self.cfg.dataset_name, split="test")
                self.val_dset = IMDB(self.tokenizer, dset["text"], dset["label"])
            elif self.cfg.dataset == "fix":        
                dset = load_dataset('json', data_files="data/IMDB_gold_gpt4o.jsonl")
                df = pd.DataFrame(dset['train'])
                df['label'] = df['label'].map(label_mapping)
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                df['label'] = df['label'].astype(int)
                dset =  datasets.Dataset.from_pandas(df)
                self.trn_dset = IMDB(self.tokenizer, dset["text"], dset["label"])
                dset = load_dataset(self.cfg.dataset_name, split="test")
                self.val_dset = IMDB(self.tokenizer, dset["text"], dset["label"])
            elif self.cfg.dataset == "origin+aug":        
                origin = load_dataset(self.cfg.dataset_name, split="train")
                dset = load_dataset('json', data_files="data/IMDB.jsonl")
                df = pd.DataFrame(dset['train'])
                df['label'] = df['label'].map(label_mapping)
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                df['label'] = df['label'].astype(int)
                dset =  datasets.Dataset.from_pandas(df)
                self.trn_dset = IMDB(self.tokenizer, dset["text"]+origin['text'], dset["label"]+origin["label"])
                dset = load_dataset(self.cfg.dataset_name, split="test")
                self.val_dset = IMDB(self.tokenizer, dset["text"], dset["label"])
            elif self.cfg.dataset == "origin+fix":        
                origin = load_dataset(self.cfg.dataset_name, split="train")
                dset = load_dataset('json', data_files="data/IMDB_gold_gpt4o.jsonl")
                df = pd.DataFrame(dset['train'])
                df['label'] = df['label'].map(label_mapping)
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                df['label'] = df['label'].astype(int)
                dset =  datasets.Dataset.from_pandas(df)
                self.trn_dset = IMDB(self.tokenizer, dset["text"]+origin['text'], dset["label"]+origin["label"])
                dset = load_dataset(self.cfg.dataset_name, split="test")
                self.val_dset = IMDB(self.tokenizer, dset["text"], dset["label"])

        if stage == "test":
            
            dset = load_dataset(self.cfg.dataset_name, split="test")
            self.tst_dset = IMDB(self.tokenizer, dset["text"], dset["label"])

    def train_dataloader(self):
        return DataLoader(
            self.trn_dset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tst_dset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.batch_size,
        )

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)
