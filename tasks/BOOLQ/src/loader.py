import pytorch_lightning as pl
import torch
import datasets
from datasets import load_dataset,concatenate_datasets

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import json
import pandas as pd
import numpy as np
class BoolQ(Dataset):
    def __init__(self, tok, question, passage, answer):
        self.tok = tok
        self.text = [[question[i],passage[i]] for i in range(len(question))]
        self.label = answer
        assert len(self.text) == len(self.label)

    def __getitem__(self, idx):
        src = self.tok(
            self.text[idx][0],self.text[idx][1], truncation=True, padding="max_length", return_tensors="pt"
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
        load_dataset(self.cfg.dataset_name, split="validation")

    def setup(self, stage: str):
        
        label_mapping = {'False':0,'True':1}
        label_mapping_original={True:1, False:0 }
        if stage == "fit":
            if self.cfg.dataset == "origin":        
                dset = load_dataset(self.cfg.dataset_name, split="train")
                df = pd.DataFrame(dset)
                df['answer'] = df['answer'].map(label_mapping_original)
                df['answer'] = df['answer'].astype(int)
                dset =  datasets.Dataset.from_pandas(df)
                self.trn_dset = BoolQ(self.tokenizer, dset["question"], dset["passage"], dset["answer"])
                dset = load_dataset(self.cfg.dataset_name, split="validation")
                df = pd.DataFrame(dset)
                df['answer'] = df['answer'].map(label_mapping_original)
                df['answer'] = df['answer'].astype(int)
                dset =  datasets.Dataset.from_pandas(df)
                self.val_dset = BoolQ(self.tokenizer, dset["question"], dset["passage"], dset["answer"])
            elif self.cfg.dataset == "aug":     
                dset = load_dataset('json', data_files="data/BoolQ.jsonl")
                df = pd.DataFrame(dset['train'])
                df['Answer'] = df['Answer'].map(label_mapping)
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                df['Answer'] = df['Answer'].astype(int)
                dset =  datasets.Dataset.from_pandas(df)
                self.trn_dset = BoolQ(self.tokenizer, dset["Question"], dset["Passage"], dset["Answer"])
                dset = load_dataset(self.cfg.dataset_name, split="validation")
                df = pd.DataFrame(dset)
                df['answer'] = df['answer'].map(label_mapping_original)
                df['answer'] = df['answer'].astype(int)
                dset =  datasets.Dataset.from_pandas(df)
                self.val_dset = BoolQ(self.tokenizer, dset["question"], dset["passage"], dset["answer"])
            elif self.cfg.dataset == "fix":        
                dset = load_dataset('json', data_files="data/BoolQ_fixed.jsonl")
                df = pd.DataFrame(dset['train'])
                df['Answer'] = df['Answer'].map(label_mapping)
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                df['Answer'] = df['Answer'].astype(int)
                dset =  datasets.Dataset.from_pandas(df)
                self.trn_dset = BoolQ(self.tokenizer, dset["Question"], dset["Passage"], dset["Answer"])
                dset = load_dataset(self.cfg.dataset_name, split="validation")
                df = pd.DataFrame(dset)
                df['answer'] = df['answer'].map(label_mapping_original)
                df['answer'] = df['answer'].astype(int)
                dset =  datasets.Dataset.from_pandas(df)
                self.val_dset = BoolQ(self.tokenizer, dset["question"], dset["passage"], dset["answer"])
            elif self.cfg.dataset == "origin+aug":        
                
                dset = load_dataset(self.cfg.dataset_name, split="train")
                df = pd.DataFrame(dset)
                df['answer'] = df['answer'].map(label_mapping_original)
                df['answer'] = df['answer'].astype(int)
                origin =  datasets.Dataset.from_pandas(df)
                dset = load_dataset('json', data_files="data/BoolQ.jsonl")
                df = pd.DataFrame(dset['train'])
                df['Answer'] = df['Answer'].map(label_mapping)
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                df['Answer'] = df['Answer'].astype(int)
                dset =  datasets.Dataset.from_pandas(df)
                self.trn_dset = BoolQ(self.tokenizer, dset["Question"]+origin["question"], dset["Passage"]+origin["passage"], dset["Answer"]+origin["answer"])
                dset = load_dataset(self.cfg.dataset_name, split="validation")
                df = pd.DataFrame(dset)
                df['answer'] = df['answer'].map(label_mapping_original)
                df['answer'] = df['answer'].astype(int)
                dset =  datasets.Dataset.from_pandas(df)
                self.val_dset = BoolQ(self.tokenizer, dset["question"], dset["passage"], dset["answer"])
            elif self.cfg.dataset == "origin+fix":        
                
                dset = load_dataset(self.cfg.dataset_name, split="train")
                df = pd.DataFrame(dset)
                df['answer'] = df['answer'].map(label_mapping_original)
                df['answer'] = df['answer'].astype(int)
                origin =  datasets.Dataset.from_pandas(df)
                dset = load_dataset('json', data_files="data/BoolQ_fixed.jsonl")
                df = pd.DataFrame(dset['train'])
                df['Answer'] = df['Answer'].map(label_mapping)
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                df['Answer'] = df['Answer'].astype(int)
                dset =  datasets.Dataset.from_pandas(df)
                self.trn_dset = BoolQ(self.tokenizer, dset["Question"]+origin["question"], dset["Passage"]+origin["passage"], dset["Answer"]+origin["answer"])
                dset = load_dataset(self.cfg.dataset_name, split="validation")
                df = pd.DataFrame(dset)
                df['answer'] = df['answer'].map(label_mapping_original)
                df['answer'] = df['answer'].astype(int)
                dset =  datasets.Dataset.from_pandas(df)
                self.val_dset = BoolQ(self.tokenizer, dset["question"], dset["passage"], dset["answer"])

        if stage == "validation":
            
            dset = load_dataset(self.cfg.dataset_name, split="validation")
            df = pd.DataFrame(dset)
            df['answer'] = df['answer'].map(label_mapping_original)
            df['answer'] = df['answer'].astype(int)
            dset =  datasets.Dataset.from_pandas(df)
            self.val_dset = BoolQ(self.tokenizer, dset["question"], dset["passage"], dset["answer"])

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
            # self.tst_dset,
            self.val_dset,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.batch_size,
        )

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)
