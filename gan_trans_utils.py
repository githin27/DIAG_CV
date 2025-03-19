import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import torch.optim as optim
from math import log10

def calculate_psnr_ssim(real_images, fake_images):
    psnr_values = []
    ssim_values = []
    fake_images = F.interpolate(fake_images, size=(real_images.shape[2], real_images.shape[3]), mode='bilinear', align_corners=False)
    real_images_np = (real_images * 0.5 + 0.5).cpu().numpy()  
    fake_images_np = (fake_images * 0.5 + 0.5).cpu().detach().numpy()
    for i in range(real_images.shape[0]):
        real_img = np.transpose(real_images_np[i], (1, 2, 0))  
        fake_img = np.transpose(fake_images_np[i], (1, 2, 0))
        if real_img.shape != fake_img.shape:
            print(f"Shape mismatch! Real: {real_img.shape}, Fake: {fake_img.shape}")
            continue
        min_dim = min(real_img.shape[0], real_img.shape[1])
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1 
        psnr_value = psnr(real_img, fake_img, data_range=1.0)
        ssim_value = ssim(real_img, fake_img, data_range=1.0, win_size=win_size, channel_axis=-1)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
    return np.mean(psnr_values), np.mean(ssim_values)


def plot_psnr_ssim(PSNR_values, SSIM_values, num_epochs, plot_path, timestamp):
    plt.figure(figsize=(10, 5))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, PSNR_values, label="PSNR")
    plt.plot(epochs, SSIM_values, label="SSIM")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("PSNR and SSIM Over Epochs")
    plt.legend()
    plt.grid(True)
    save_file = os.path.join(plot_path, f"psnr_ssim_plot_{timestamp}.png")
    plt.savefig(save_file)
    plt.close()
    #print(f"[INFO] PSNR/SSIM plot saved to {save_file}")

def plot_precision_recall_f1(Precision_values, Recall_values, F1_values, num_epochs, plot_path, timestamp):
    plt.figure(figsize=(10, 5))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, Precision_values, label="Precision")
    plt.plot(epochs, Recall_values, label="Recall")
    plt.plot(epochs, F1_values, label="F1-score")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1-score Over Epochs")
    plt.legend()
    plt.grid(True)
    save_file = os.path.join(plot_path, f"precision_recall_f1_plot_{timestamp}.png")
    plt.savefig(save_file)
    plt.close()
    #print(f"[INFO] Precision/Recall/F1 plot saved to {save_file}")

def plot_loss(G_losses, D_losses, num_epochs, plot_path, timestamp):
    plt.figure(figsize=(10, 5))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, G_losses, label="Generator Loss")
    plt.plot(epochs, D_losses, label="Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    save_file = os.path.join(plot_path, f"loss_plot_{timestamp}.png")
    plt.savefig(save_file)
    plt.close()
    #print(f"[INFO] Loss plot saved to {save_file}")


class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for file_name in os.listdir(label_dir):
                    file_path = os.path.join(label_dir, file_name)
                    self.images.append(file_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Multi-Head Self-Attention Module
def scaled_dot_product_attention(q, k, v):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    attn = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.scale = (in_dim // heads) ** 0.5
        self.query = nn.Linear(in_dim, in_dim, bias=False)
        self.key = nn.Linear(in_dim, in_dim, bias=False)
        self.value = nn.Linear(in_dim, in_dim, bias=False)
        self.out = nn.Linear(in_dim, in_dim, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_output = scaled_dot_product_attention(q, k, v)
        out = self.out(attn_output).permute(0, 2, 1).view(B, C, H, W)
        return out

# Generator with Self-Attention
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            SelfAttention(ngf * 8),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            SelfAttention(ngf * 4),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Discriminator with Global Average Pooling
class Discriminator(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(ndf),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(ndf * 2),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        out = self.main(input)
        return out.view(input.size(0), 1)

def unnormalize(img_tensor):
    return img_tensor * 0.5 + 0.5

def compute_ssim_test(img1, img2):
    img1_np = img1.permute(1,2,0).cpu().numpy()
    img2_np = img2.permute(1,2,0).cpu().numpy()
    return ssim(img1_np, img2_np, multichannel=True, data_range=1.0)

def compute_psnr_test(img1, img2):
    mse = ((img1 - img2)**2).mean().item()
    if mse == 0:
        return 100  
    return 10 * log10(1.0 / mse)


class Generator_test(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input Z: (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

def optimize_latent_test(netG, real_img, nz=100, num_iter=30, lr=0.01, device='cpu'):
    criterion = nn.MSELoss()
    
    # Initialize z randomly
    z = torch.randn(1, nz, 1, 1, device=device, requires_grad=True)
    optimizer_z = optim.Adam([z], lr=lr)

    for _ in range(num_iter):
        optimizer_z.zero_grad()
        recon = netG(z)
        loss = criterion(recon, real_img)
        loss.backward()
        optimizer_z.step()

    with torch.no_grad():
        final_recon = netG(z)
    return z, final_recon

