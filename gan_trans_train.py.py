import os
import sys
import datetime
import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import *

def main():
    parser = argparse.ArgumentParser(description="Training script for anomaly detection.")
    default_dataset_path = "./default_data/MVTec_dataset/wood/train/"
    #default_dataset_path = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "MVTec-Anomaly-Detection_MindColab", "MVTec_dataset", "wood", "train"))
    parser.add_argument("dataset_path", nargs="?", default=default_dataset_path,
                        help="Path to training dataset. Default: %(default)s")
    parser.add_argument("num_epochs", nargs="?", type=int, default=100,
                        help="Number of training epochs. Default: %(default)s")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    data_path = args.dataset_path
    num_epochs = args.num_epochs
    print(f"[INFO] Dataset Path: {data_path}")

    output_folder = f"Train_Output_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"[INFO] Output Folder : {output_folder}")

    save_path = os.path.join(output_folder, f"trained_model_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    print(f"[INFO] Model save path : {save_path}")

    log_file_path = os.path.join(output_folder, f"training_log_{timestamp}.txt")
    print(f"[INFO] Log save path : {log_file_path}")

    plot_path = os.path.join(output_folder, f"plot_imgs_{timestamp}")
    os.makedirs(plot_path, exist_ok=True)
    print(f"[INFO] Plot save path : {plot_path}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    train_dataset = MVTecDataset(root_dir=data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Initialize models
    nz = 100
    nc = 3
    ngf = 64
    ndf = 64
    ngpu = 1
    netG = Generator(nz, ngf, nc, ngpu).to(device)
    netD = Discriminator(isize=256, nz=nz, nc=nc, ndf=ndf, ngpu=ngpu).to(device)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.00005, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    G_losses = []
    D_losses = []
    PSNR_values = []
    SSIM_values = []
    Precision_values = []
    Recall_values = []
    F1_values = []

    with open(log_file_path, 'w') as f:
        for epoch in range(num_epochs):
            epoch_loss_G = 0
            epoch_loss_D = 0
            epoch_psnr = 0
            epoch_ssim = 0
            all_real_labels = []
            all_fake_labels = []
            all_real_preds = []
            all_fake_preds = []

            for i, data in enumerate(train_loader, 0):
                netD.zero_grad()
                real_images = data.to(device)
                batch_size = real_images.size(0)
                if batch_size == 0:
                    continue
                real_images += 0.1 * torch.randn_like(real_images).to(device)
                label = torch.full((batch_size, 1), 0.9, dtype=torch.float, device=device)
                real_pred = netD(real_images).detach().cpu().numpy()
                all_real_preds.extend(real_pred)
                all_real_labels.extend(np.ones(batch_size))  # Label 1 for real images
                errD_real = criterion(netD(real_images), label)
                errD_real.backward()
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = netG(noise)
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = netG(noise) + 0.1 * torch.randn_like(fake).to(device)
                label.fill_(0.1)
                fake_pred = netD(fake.detach()).detach().cpu().numpy()
                all_fake_preds.extend(fake_pred)
                all_fake_labels.extend(np.zeros(batch_size))  # Label 0 for fake images
                errD_fake = criterion(netD(fake.detach()), label)
                errD_fake.backward()
                optimizerD.step()
                netG.zero_grad()
                label.fill_(0.9)
                errG = criterion(netD(fake), label)
                errG.backward()
                optimizerG.step()
                epoch_loss_G += errG.item()
                epoch_loss_D += errD_real.item() + errD_fake.item()

                psnr_val, ssim_val = calculate_psnr_ssim(real_images, fake)
                epoch_psnr += psnr_val
                epoch_ssim += ssim_val

                progress_msg = f"[{epoch+1}/{num_epochs}][{i+1}/{len(train_loader)}]"
                sys.stdout.write("\r" + progress_msg)
                sys.stdout.flush()

            all_real_preds = np.array(all_real_preds).flatten()
            all_fake_preds = np.array(all_fake_preds).flatten()
            all_preds = np.concatenate([all_real_preds, all_fake_preds])
            all_labels = np.concatenate([all_real_labels, all_fake_labels])
            binary_preds = (all_preds >= 0.5).astype(int)

            precision = precision_score(all_labels, binary_preds, zero_division=0)
            recall = recall_score(all_labels, binary_preds)
            f1 = f1_score(all_labels, binary_preds)

            Precision_values.append(precision)
            Recall_values.append(recall)
            F1_values.append(f1)

            avg_G_loss = epoch_loss_G / len(train_loader)
            avg_D_loss = epoch_loss_D / len(train_loader)
            avg_psnr = epoch_psnr / len(train_loader)
            avg_ssim = epoch_ssim / len(train_loader)

            G_losses.append(avg_G_loss)
            D_losses.append(avg_D_loss)
            PSNR_values.append(avg_psnr)
            SSIM_values.append(avg_ssim)

            log_msg = (f" Loss G: {avg_G_loss:.4f}, Loss D: {avg_D_loss:.4f}, "
                       f"PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, "
                       f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
            print(log_msg)
            f.write(log_msg + '\n')

    torch.save(netG, os.path.join(save_path, "generator_model.pth"))
    torch.save(netD, os.path.join(save_path, "discriminator_model.pth"))
    print(f"[INFO] Models saved in {save_path}")

    plot_psnr_ssim(PSNR_values, SSIM_values, num_epochs, plot_path, timestamp)
    plot_precision_recall_f1(Precision_values, Recall_values, F1_values, num_epochs, plot_path, timestamp)
    plot_loss(G_losses, D_losses, num_epochs, plot_path, timestamp)

    print(f"[INFO] Training logs saved in {log_file_path}")
    print(f"[INFO] Plots saved in {plot_path}")
    print(f"[INFO] Training completed!")

if __name__ == "__main__":
    main()
