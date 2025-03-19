import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, ff_dim=512, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x

class ResNetTransformerModel(nn.Module):
    def __init__(self, num_classes=4, weights_path="resnet50_weights.pth"):
        super(ResNetTransformerModel, self).__init__()
        resnet = models.resnet50(weights=None)
        # Load weights from the local file.
        resnet.load_state_dict(torch.load(weights_path, map_location="cpu"))
        # Remove the final avgpool and fc layers.
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        # Reduce channels from 2048 to 256.
        self.channel_reducer = nn.Conv2d(2048, 256, kernel_size=1)
        # Add Transformer block.
        self.transformer = TransformerBlock(embed_dim=256, num_heads=4)
        # Final classifier.
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.feature_extractor(x)  # (batch, 2048, H, W)
        x = self.channel_reducer(x)      # (batch, 256, H, W)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(2, 0, 1)  # (seq_len, batch, 256)
        x = self.transformer(x)
        x = x.mean(dim=0)  # (batch, 256)
        x = self.fc(x)
        return x
        
class MVTecDataset_test(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Class mapping for wood test categories:
        class_mapping = {
            "good": 0,
            "color": 1,
            "combined": 2,
            "hole": 3,
            "liquid": 4,
            "scratch": 5
        }

        for class_name, label in class_mapping.items():
            class_dir = os.path.join(root_dir, "test", class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        class_mapping = {"good": 0}
        
        for class_name, label in class_mapping.items():
            class_dir = os.path.join(root_dir, "train", class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def plot_loss(all_epochs, train_losses, val_epochs, val_losses, timestamp, plot_path):
    plt.figure(figsize=(10, 5))
    plt.plot(all_epochs, train_losses, marker='o', label="Train Loss")
    plt.plot(val_epochs, val_losses, marker='o', label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.grid(True)
    loss_plot_file = os.path.join(plot_path, f"loss_vs_epoch_{timestamp}.png")
    plt.savefig(loss_plot_file)
    plt.close()
    print(f"[INFO] Loss plot saved in {loss_plot_file}")

def plot_accuracy(all_epochs, train_accuracies, val_epochs, val_accuracies, timestamp, plot_path):
    plt.figure(figsize=(10, 5))
    plt.plot(all_epochs, train_accuracies, marker='o', label="Train Accuracy")
    plt.plot(val_epochs, val_accuracies, marker='o', label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.grid(True)
    acc_plot_file = os.path.join(plot_path, f"accuracy_vs_epoch_{timestamp}.png")
    plt.savefig(acc_plot_file)
    plt.close()
    print(f"[INFO] Accuracy plot saved in {acc_plot_file}")

def plot_train_metrics(all_epochs, precisions, recalls, f1s, timestamp, plot_path):
    plt.figure(figsize=(10, 5))
    plt.plot(all_epochs, precisions, marker='o', label="Train Precision")
    plt.plot(all_epochs, recalls, marker='o', label="Train Recall")
    plt.plot(all_epochs, f1s, marker='o', label="Train F1")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.title("Training Metrics over Epochs")
    plt.legend()
    plt.grid(True)
    metrics_plot_file = os.path.join(plot_path, f"train_metrics_vs_epoch_{timestamp}.png")
    plt.savefig(metrics_plot_file)
    plt.close()
    print(f"[INFO] Training metrics plot saved in {metrics_plot_file}")
