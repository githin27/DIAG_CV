import torch
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torchvision import transforms
from utils import ResNetTransformerModel, MVTecDataset_test

def load_model(model_path, device):
    """Load the trained model."""
    model = torch.load(model_path, map_location=device)  # Load entire model
    model.to(device)
    model.eval()
    return model

def evaluate_model(model_path, test_data_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_dir, f"Test_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    print(f"[INFO] Saving outputs to: {output_folder}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = MVTecDataset_test(test_data_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    #print(f"[INFO] Number of test samples: {len(test_dataset)}")
    
    if len(test_dataset) == 0:
        print("[ERROR] Test dataset is empty. Check the dataset path and structure.")
        return
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if not all_preds:
        print("[ERROR] No predictions made. Check the test dataset and model.")
        return
    
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds)
    class_names = ["good", "color", "combined", "hole", "liquid", "scratch"]
    
    log_file_path = os.path.join(output_folder, "test_metrics.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Precision: {precision:.4f}\n")
        log_file.write(f"Recall: {recall:.4f}\n")
        log_file.write(f"F1 Score: {f1:.4f}\n")
        print(f"[INFO] Log file saved to {log_file_path}")
    
    if cm.size > 0:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        cm_file_path = os.path.join(output_folder, "confusion_matrix.png")
        plt.savefig(cm_file_path)
        plt.close()
        print(f"[INFO] Confusion matrix saved to {cm_file_path}")
    else:
        print(f"[WARNING] Confusion matrix is empty. No results to plot.")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test script for anomaly detection.")
    parser.add_argument("--model_path", type=str, default="default_data/model.pth", help="Path to the trained model.")
    parser.add_argument("--dataset_path", type=str, default="default_data/MVTec_dataset/wood/test", help="Path to the test dataset.")
    parser.add_argument("--output_dir", type=str, default="Test_output", help="Directory to save test results.")
    args = parser.parse_args()
    evaluate_model(args.model_path, args.dataset_path, args.output_dir)
