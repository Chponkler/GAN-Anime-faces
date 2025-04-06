# Anime Character Generation with GAN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/your-repo/blob/main/Anime_GAN.ipynb)

Генеративно-состязательная сеть (GAN) для создания аниме-персонажей с использованием PyTorch и Google Colab.

## Особенности

- Архитектура DCGAN с Self-Attention слоями
- Реализация WGAN-GP с Gradient Penalty
- Поддержка GPU через Google Colab (T4/V100)
- Аугментация данных для улучшения качества
- Автоматическое сохранение весов и примеров

## Требования

- Google Аккаунт
- Google Colab (среда выполнения с GPU)
- PyTorch 2.0+
- Данные: 1000+ аниме-изображений (64x64px)

## Быстрый старт

1. **Подготовка данных**
   - Создайте папку в Google Drive: `MyDrive/dataset/data`
   - Загрузите изображения в формате JPG/PNG (пример: 1000+ изображений 64x64px)

2. **Запуск в Google Colab**
   - Откройте [ноутбук в Colab](https://colab.research.google.com/)
   - Выберите `Runtime` → `Change runtime type` → **GPU** (T4 или V100)
   - Скопируйте [предоставленный код](#код) в ячейку Colab
   - Запустите все ячейки (Ctrl+F9)

3. **Мониторинг обучения**
   - Веса моделей сохраняются в `gan_results` каждые 10 эпох
   - Примеры генерации сохраняются как `epoch_XXX.png`

## Архитектура модели

### Генератор
```python
Generator(
  (main): Sequential(
    (0): ConvTranspose2d(128, 512, ...)
    (1): BatchNorm2d(512)
    (2): ReLU()
    ...
    (13): Tanh()
  )
)
```
Гиперпараметры

-Latent dim	128

-Batch size	64

-Learning rate	0.0002

-Epochs	200

-Lambda GP	10


![Моя фотография](C:\Users\Fedor\Desktop/40ep.jpg)
