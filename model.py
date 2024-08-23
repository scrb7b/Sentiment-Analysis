import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from build_dataset import ToxicCommentsDataset, get_datasets

LEARNING_RATE = 2e-5

class ToxicCommentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(ToxicCommentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output
        output = self.drop(pooled_output)
        return self.out(output)


model = ToxicCommentClassifier(n_classes=3)
model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

