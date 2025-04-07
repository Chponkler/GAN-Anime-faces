import torch
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np

# Конфигурация
latent_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Self-Attention слой
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)

        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


# Генератор (такой же как в обучении)
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


# Загрузка обученной модели
def load_generator(model_path):
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    return generator


# Генерация изображения
def generate_image(generator):
    with torch.no_grad():
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        generated_image = generator(noise).cpu()
        # Преобразование из [-1, 1] в [0, 1]
        generated_image = (generated_image + 1) / 2
        return generated_image[0]


# Преобразование тензора в изображение для Tkinter
def tensor_to_tkimage(tensor):
    # Преобразование тензора в numpy array
    image = tensor.permute(1, 2, 0).numpy()
    image = (image * 255).astype('uint8')
    # Создание PIL Image
    pil_image = Image.fromarray(image)
    # Преобразование для Tkinter
    tk_image = ImageTk.PhotoImage(pil_image)
    return tk_image


# GUI приложение
class GANApp:
    def __init__(self, root, generator):
        self.root = root
        self.generator = generator
        self.root.title("Генератор изображений")
        self.root.geometry("400x500")

        # Изображение
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=20)

        # Кнопка генерации
        self.generate_btn = tk.Button(
            root,
            text="Сгенерировать изображение",
            command=self.generate_and_show,
            height=2,
            width=20,
            font=('Arial', 12)
        )
        self.generate_btn.pack(pady=20)

        # Статус
        self.status_label = tk.Label(root, text="Готов к генерации", fg="green")
        self.status_label.pack(pady=10)

    def generate_and_show(self):
        try:
            self.status_label.config(text="Генерация...", fg="blue")
            self.root.update()

            # Генерация изображения
            generated_tensor = generate_image(self.generator)
            tk_image = tensor_to_tkimage(generated_tensor)

            # Обновление изображения
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image

            self.status_label.config(text="Изображение сгенерировано", fg="green")
        except Exception as e:
            self.status_label.config(text=f"Ошибка: {str(e)}", fg="red")


# Основная функция
def main():
    # Инициализация GUI
    root = tk.Tk()

    # Запрос пути к модели
    model_path = filedialog.askopenfilename(
        title="Выберите файл с обученной моделью генератора",
        filetypes=[("PyTorch Model", "*.pth")]
    )

    if not model_path:
        print("Модель не выбрана. Выход.")
        return

    # Загрузка модели
    try:
        generator = load_generator(model_path)
        print("Модель успешно загружена!")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return

    # Запуск приложения
    app = GANApp(root, generator)
    root.mainloop()


if __name__ == "__main__":
    main()
