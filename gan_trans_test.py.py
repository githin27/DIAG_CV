import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import datetime
from utils import MVTecDataset, Generator
def unnormalize(img_tensor):
    return img_tensor * 0.5 + 0.5

def main():
    default_model_path = "./default_data/generator_model.pth"
    default_dataset_path = "./default_data/MVTec_dataset/wood/test"

    parser = argparse.ArgumentParser(description="Test script for image reconstruction using a trained generator.")
    parser.add_argument("--model_path", type=str, default=default_model_path,
                        help="Path to the trained generator model (e.g., generator_model.pth)")
    parser.add_argument("--dataset_path", type=str, default=default_dataset_path,
                        help="Path to the test dataset. Default points to './default_test/MVTec_dataset/test/'.")
    parser.add_argument("--num_images", type=int, default=5,
                        help="Number of test images to process. Default: 5")
    parser.add_argument("--num_iter", type=int, default=30,
                        help="Number of iterations for latent optimization. Default: 30")
    parser.add_argument("--nz", type=int, default=100,
                        help="Dimension of latent vector. Default: 100")

    args, unknown = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"Test_Output_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"[INFO] Output Folder : {output_folder}")
    
    netG = torch.load(args.model_path, map_location=device)
    netG.eval()  # set to evaluation mode

    print(f"[INFO] Test Dataset Path: {args.dataset_path}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_dataset = MVTecDataset(root_dir=args.dataset_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    mse_loss = nn.MSELoss()

    images_to_plot = []
    reconstructions_to_plot = []
    count = 0

    for data in test_loader:
        x = data.to(device)  

        z = torch.randn(1, args.nz, 1, 1, device=device, requires_grad=True)
        optimizer_z = optim.Adam([z], lr=0.01)

        for i in range(args.num_iter):
            optimizer_z.zero_grad()
            reconstructed = netG(z)
            loss = mse_loss(reconstructed, x)
            loss.backward()
            optimizer_z.step()

        with torch.no_grad():
            reconstruction = netG(z)

        images_to_plot.append(x.squeeze(0).cpu())
        reconstructions_to_plot.append(reconstruction.squeeze(0).cpu())
        count += 1

        if count >= args.num_images:
            break

    num_images = len(images_to_plot)
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))

    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_images):
        orig = unnormalize(images_to_plot[i]).permute(1, 2, 0).numpy()
        recon = unnormalize(reconstructions_to_plot[i]).permute(1, 2, 0).numpy()

        diff = np.abs(orig - recon)
        
        # --- Original ---
        axes[i, 0].imshow(np.clip(orig, 0, 1))
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        # --- Reconstructed ---
        axes[i, 1].imshow(np.clip(recon, 0, 1))
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].axis("off")

        # --- Difference / Anomaly Highlight ---
        diff_gray = diff.mean(axis=-1)
        diff_gray = np.clip(diff_gray * 5.0, 0, 1)
        im = axes[i, 2].imshow(diff_gray, cmap="jet")
        axes[i, 2].set_title("Difference")
        axes[i, 2].axis("off")

    plt.tight_layout()
    save_file = os.path.join(output_folder, f"test_plot_{timestamp}.png")
    plt.savefig(save_file)
    print(f"[INFO] Plot saved to {save_file}")
    plt.show()

if __name__ == "__main__":
    main()
