#!/bin/bash
# ===== RunPod Setup Script =====
# Автоматическая установка всех зависимостей

echo "🚀 Starting RunPod setup..."

# Обновляем pip
pip install --upgrade pip

# Устанавливаем зависимости
echo "📦 Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
pip install datasets huggingface-hub

echo "✅ Dependencies installed!"

# Авторизация в Hugging Face (опционально для LLaMA)
# huggingface-cli login --token YOUR_TOKEN_HERE

echo "🎯 Setup complete! Ready to train."
echo ""
echo "Запустите обучение:"
echo "  python train.py"

