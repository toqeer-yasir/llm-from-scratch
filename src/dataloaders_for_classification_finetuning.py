import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class SpamDataset(Dataset):
    def __init__(self, csv_df, tokenizer, max_length=None, pad_token_id=50256):
        self.df = pd.read_csv(csv_df)
        self.encoded_text = [
            tokenizer.encode(text) for text in self.df['Text']
        ]

        if max_length is None:
            self.max_length = max([len(text) for text in self.encoded_text])
        else:
            self.max_length = max_length

            self.encoded_text = [
                text[:self.max_length] for text in self.encoded_text
                ]

        self.encoded_text = [
            text + [pad_token_id] * (self.max_length - len(text)) for text in self.encoded_text
        ]


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        encoded = self.encoded_text[idx]
        label = self.df['Label'][idx]
        encoded = torch.tensor(encoded, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        return encoded, label
    

tokenizer = tiktoken.get_encoding('gpt2')

train_df = '/content/drive/MyDrive/llm_from_scratch/datasets/train_df.csv'
validation_df = '/content/drive/MyDrive/llm_from_scratch/datasets/validation_df.csv'
test_df = '/content/drive/MyDrive/llm_from_scratch/datasets/test_df.csv'

train_dataset = SpamDataset(train_df, tokenizer)
validation_dataset = SpamDataset(validation_df, tokenizer)
test_dataset = SpamDataset(test_df, tokenizer)

batch_size = 8

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)