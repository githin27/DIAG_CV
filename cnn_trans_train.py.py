import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import os
import sys
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import *

def main(dataset_path, num_epochs):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"Output_train_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"[INFO] Output Folder : {output_folder}")
    
    save_path = os.path.join(output_folder, f"trained_model_{timestamp}.pth")
    log_file_path = os.path.join(output_folder, f"training_log_{timestamp}.txt")
    plot_path = os.path.join(output_folder, f"plot_imgs_{timestamp}")
    os.makedirs(plot_path, exist_ok=True)
    print(f"[INFO] Plot save path : {plot_path}")
    
    log_file = open(log_file_path, "w")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print(f"[INFO] Dataset Path: {dataset_path}")
    log_file.write(f"[INFO] Dataset Path: {dataset_path}\n")
    
    full_dataset = MVTecDataset(dataset_path, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = ResNetTransformerModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    class_weights = torch.tensor([1.0, 1.5, 1.5, 1.2]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    epochs = num_epochs

    epoch_losses = []
    epoch_accuracies = []
    epoch_precisions = []
    epoch_recalls = []
    epoch_f1s = []
    
    val_losses = []
    val_accuracies = []
    
    best_val_accuracy = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_train_preds.extend(predicted.cpu().numpy().tolist())
            all_train_labels.extend(labels.cpu().numpy().tolist())
            
            sys.stdout.write(f"\r[{epoch+1}/{epochs}]")
            sys.stdout.flush()
        
        train_loss = running_loss / total
        train_accuracy = 100 * correct / total
        train_precision = precision_score(all_train_labels, all_train_preds, average="macro", zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, average="macro", zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds, average="macro", zero_division=0)
        
        epoch_losses.append(train_loss)
        epoch_accuracies.append(train_accuracy)
        epoch_precisions.append(train_precision)
        epoch_recalls.append(train_recall)
        epoch_f1s.append(train_f1)
        
        # Validation loop.
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                all_val_preds.extend(predicted.cpu().numpy().tolist())
                all_val_labels.extend(labels.cpu().numpy().tolist())
        avg_val_loss = val_loss / val_total
        avg_val_accuracy = 100 * val_correct / val_total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)
        
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            best_epoch = epoch + 1
            best_model_path = os.path.join(output_folder, f"best_model_{timestamp}.pth")
            torch.save(model, best_model_path)
        
        log_msg = (f" Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, "
                   f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.2f}%")
        print(log_msg)
        log_file.write(log_msg + "\n")
        
        scheduler.step()
    
    print("[INFO] Training complete!")
    log_file.write("[INFO] Training complete!\n")
    log_file.write(f"[INFO] Best Validation Accuracy: {best_val_accuracy:.2f}% at Epoch {best_epoch}\n")
    log_file.close()

    torch.save(model, save_path)
    print(f"[INFO] Final model saved in {save_path}")
    print(f"[INFO] Logs saved in {log_file_path}")
    
    all_epochs = list(range(1, epochs + 1))
    
    plot_loss(all_epochs, epoch_losses, all_epochs, val_losses, timestamp, plot_path)
    plot_accuracy(all_epochs, epoch_accuracies, all_epochs, val_accuracies, timestamp, plot_path)
    plot_train_metrics(all_epochs, epoch_precisions, epoch_recalls, epoch_f1s, timestamp, plot_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training script for anomaly detection.")
    
    default_dataset_path = "./default_data/MVTec_dataset/wood/train/"
    parser.add_argument("dataset_path", nargs="?", default=default_dataset_path,
                        help="Path to training dataset. Default: %(default)s")
    parser.add_argument("num_epochs", nargs="?", type=int, default=5,
                        help="Number of training epochs. Default: %(default)s")
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    epochs = args.num_epochs

    main(dataset_path, epochs)
