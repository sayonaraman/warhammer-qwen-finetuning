#!/bin/bash
# ===== RunPod QuickStart =====
# Один скрипт для полной автоматизации

set -e  # Остановка при ошибке

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  🚀 RunPod QuickStart - Qwen 2.5 7B Fine-tuning      ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Проверка GPU
echo "🔍 Проверка GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Установка зависимостей
echo "📦 [1/4] Установка зависимостей..."
pip install -q --upgrade pip
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -q --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
pip install -q datasets huggingface-hub
echo "   ✅ Зависимости установлены"
echo ""

# Проверка файлов
echo "📁 [2/4] Проверка файлов проекта..."
if [ ! -f "config.py" ]; then
    echo "   ❌ config.py не найден!"
    exit 1
fi
if [ ! -f "train_runpod.py" ]; then
    echo "   ❌ train_runpod.py не найден!"
    exit 1
fi
if [ ! -d "input_data" ] && [ ! -d "warhammer_dataset" ]; then
    echo "   ❌ Нужна папка input_data/ или warhammer_dataset/!"
    exit 1
fi
echo "   ✅ Все файлы на месте"
echo ""

# Создание датасета (если нужно)
if [ ! -d "warhammer_dataset" ]; then
    echo "🔨 [3/4] Создание датасета..."
    python prepare_dataset.py
    echo "   ✅ Датасет создан"
else
    echo "✅ [3/4] Датасет уже существует"
fi
echo ""

# Запуск обучения
echo "🚀 [4/4] ЗАПУСК ОБУЧЕНИЯ..."
echo "════════════════════════════════════════════════════════"
echo ""

python train_runpod.py

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  ✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!                               ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Модель сохранена в: fine_tuned_model/"
echo ""
echo "💾 Чтобы скачать модель:"
echo "   1. В Jupyter: ПКМ на папку → Download as Archive"
echo "   2. Или: zip -r fine_tuned_model.zip fine_tuned_model/"
echo ""
echo "⚠️  НЕ ЗАБУДЬТЕ ОСТАНОВИТЬ POD в RunPod Dashboard!"
echo ""

