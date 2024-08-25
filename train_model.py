import torch
from build_dataset import get_datasets
from model import model, criterion, optimizer

train_loader, val_loader = get_datasets()

EPOCHS = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(EPOCHS):

    correct_predictions = 0
    run_loss = 0
    pre = 0
    for i, data in enumerate(train_loader):

        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        pre += torch.sum(labels)
        run_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f'Epoch {epoch + 1}/{EPOCHS} loss {run_loss:.3f}')
        print(f'Corect predict {correct_predictions}/{pre}')

torch.save(model.state_dict(), 'Sentiment_Bert_3_epochs.pth')
