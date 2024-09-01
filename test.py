import torch
from sklearn.metrics import accuracy_score
from build_dataset import get_datasets
from model import model

train_loader, val_loader = get_datasets()

EPOCHS = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('Sentiment_Bert_1_epochs.pth'))
y_pred = []
y_true = []

model = model.eval()

with torch.no_grad():
    for d in val_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy}')