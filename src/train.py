import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import EmotionRecognitionModel
from dataloader import get_dataloader
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10):
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        

        train_loss = running_loss / total
        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_dir = os.path.join(PROJECT_ROOT, "experiments", "checkpoints")
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            print(f"Nouveau meilleur modèle sauvegardé ! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping après {epoch+1} époques (pas d'amélioration depuis {patience} époques)")
            break
            
        scheduler.step()

    print("Training complete. Best validation accuracy: {:.4f}".format(best_val_acc))

def evaluate(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

if __name__ == "__main__":
    batch_size = 32 
    num_epochs = 120
    learning_rate = 1e-4
    num_classes = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_loader, val_loader, test_loader = get_dataloader(
        data_dir=os.path.join(PROJECT_ROOT, "archive"),
        image_size=48,
        batch_size=batch_size
    )   

    model = EmotionRecognitionModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=num_epochs)