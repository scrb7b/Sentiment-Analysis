import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

MAX_LEN = 128
BATCH_SIZE = 16

class ToxicCommentsDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_len):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {'neutral': 0, 'positive': 1, 'negative': 2}

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        label = self.label_map[self.labels[idx]]

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'comment_text': comment,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_datasets():
    data = pd.read_csv('data/data.csv')
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = ToxicCommentsDataset(
        comments=train_data['Sentence'].to_numpy(),
        labels=train_data['Sentiment'].to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    val_dataset = ToxicCommentsDataset(
        comments=val_data['Sentence'].to_numpy(),
        labels=val_data['Sentiment'].to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

if __name__ == '__main__':
    get_datasets()