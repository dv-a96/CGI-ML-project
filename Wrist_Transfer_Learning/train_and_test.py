import torch
from torchvision.models import InceptionOutputs
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import os
import pandas as pd

def train(model, train_loader, criterion, optimizer, number_of_epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    batch_losses = []

    for epoch in range(number_of_epochs):
        train_loss = 0
        correct = 0
        total = 0
        for x_data, y_label in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{number_of_epochs}"):
            x_data, y_label = x_data.to(device), y_label.to(device)

            optimizer.zero_grad()

            y_pred = model(x_data)

            if isinstance(y_pred, InceptionOutputs):
                y_pred = y_pred.logits

            loss = criterion(y_pred, y_label)
            batch_losses.append(loss.item())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            predicted = y_pred.argmax(dim=1)
            total += y_label.size(0)
            correct += predicted.eq(y_label).sum().item()

        train_loss = train_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch: {epoch + 1} \tTraining Loss: {train_loss:.6f} \tAccuracy: {accuracy:.2f}%')

    return model

def test(model, test_loader, criterion, batch_size, data_df):
    test_loss = 0
    model.eval()
    correct = 0

    fp_samples = []
    fn_samples = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for index, (x_data, y_label) in enumerate(test_loader):
            x_data, y_label = x_data.to(device), y_label.to(device)

            y_pred = model(x_data)

            if isinstance(y_pred, InceptionOutputs):
                y_pred = y_pred.logits

            loss = criterion(y_pred, y_label)
            test_loss += loss.item()

            predicted = y_pred.argmax(dim=1)

            correct += predicted.eq(y_label).sum().item()

            for i in range(x_data.size(0)):  # Changed from batch_size to x_data.size(0)
                if y_label[i] == 0 and predicted[i] == 1:
                    fp_samples.append(data_df.iloc[index * batch_size + i].to_dict())
                elif y_label[i] == 1 and predicted[i] == 0:
                    fn_samples.append(data_df.iloc[index * batch_size + i].to_dict())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    # Save false positives and false negatives to CSV files
    fp_df = pd.DataFrame(fp_samples)
    fn_df = pd.DataFrame(fn_samples)
    fp_df.to_csv('false_positives.csv', index=False)
    fn_df.to_csv('false_negatives.csv', index=False)

    return accuracy
