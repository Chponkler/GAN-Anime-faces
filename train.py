import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from google.colab import drive

# Подключение Google Drive
drive.mount('/content/drive')

# Конфигурация
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128
batch_size = 64
epochs = 200
save_dir = '/content/drive/MyDrive/gan_results'
data_dir = '/content/drive/MyDrive/dataset/data21/data'

# Создание директорий
os.makedirs(save_dir, exist_ok=True)

# Аугментации
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
    transforms.RandomAdjustSharpness(1.5, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Кастомный датасет
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir)
                          if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

# Self-Attention слой
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H*W).permute(0,2,1)
        k = self.key(x).view(B, -1, H*W)
        v = self.value(x).view(B, -1, H*W)

        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0,2,1)).view(B,C,H,W)
        return self.gamma * out + x

# Генератор
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            SelfAttention(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Дискриминатор
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(256),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.main(x).view(-1)

# Gradient Penalty для WGAN-GP
def gradient_penalty(D, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = D(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

# Инициализация моделей
G = Generator().to(device)
D = Discriminator().to(device)

# Инициализация весов
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

G.apply(weights_init)
D.apply(weights_init)

# Оптимизаторы
optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.9))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.9))

# Загрузка датасета
dataset = CustomDataset(root_dir=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Фиксированный шум для визуализации
fixed_noise = torch.randn(8, latent_dim, 1, 1, device=device)
lambda_gp = 10

# Цикл обучения
for epoch in range(epochs):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for i, (real_imgs, _) in enumerate(progress_bar):
        real_imgs = real_imgs.to(device)
        current_batch_size = real_imgs.size(0)

        # Обучение Discriminator
        optimizer_D.zero_grad()

        # Генерация фейковых изображений
        z = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
        fake_imgs = G(z)

        # WGAN-GP Loss
        real_output = D(real_imgs)
        fake_output = D(fake_imgs.detach())
        gp = gradient_penalty(D, real_imgs, fake_imgs, device)
        d_loss = -torch.mean(real_output) + torch.mean(fake_output) + lambda_gp * gp
        d_loss.backward()
        optimizer_D.step()

        # Обучение Generator (каждые 5 итераций)
        if i % 5 == 0:
            optimizer_G.zero_grad()
            new_z = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
            new_fake_imgs = G(new_z)
            fake_output = D(new_fake_imgs)
            g_loss = -torch.mean(fake_output)
            g_loss.backward()
            optimizer_G.step()

        progress_bar.set_postfix({
            "G_loss": g_loss.item(),
            "D_loss": d_loss.item(),
            "GP": gp.item()
        })

    # Сохранение моделей и примеров
    if (epoch + 1) % 10 == 0 or epoch == 0:
        torch.save(G.state_dict(), os.path.join(save_dir, f'generator_epoch_{epoch+1}.pth'))
        torch.save(D.state_dict(), os.path.join(save_dir, f'discriminator_epoch_{epoch+1}.pth'))

        with torch.no_grad():
            generated = G(fixed_noise)
            save_image(
                generated,
                os.path.join(save_dir, f'epoch_{epoch+1}.png'),
                nrow=4,
                normalize=True
            )

print("Обучение завершено!")
