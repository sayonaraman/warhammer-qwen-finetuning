"""
Прямое скачивание Qwen 2.5 7B через snapshot_download
Обходит проблему с AutoModelForCausalLM
"""

from huggingface_hub import snapshot_download
import os

model_name = "Qwen/Qwen2.5-7B-Instruct"
local_dir = "qwen2.5-7b-instruct"

print(f"Скачивание модели: {model_name}")
print(f"Это займет время (~15GB)...")
print(f"Сохранение в: {local_dir}\n")

try:
    # Прямое скачивание всех файлов модели
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        resume_download=True,  # Можно прерывать и продолжать
        local_dir_use_symlinks=False  # Для Windows
    )
    
    print("\n" + "="*60)
    print("ГОТОВО!")
    print("="*60)
    print(f"Модель сохранена: {os.path.abspath(local_dir)}")
    print("\nТеперь в train.py используйте:")
    print(f'    model_name="{local_dir}"')
    print("="*60)
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()

