# ===== КРИТИЧНО ДЛЯ WINDOWS: Защита от multiprocessing =====
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Отключаем параллелизм токенизатора
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
if __name__ == '__main__':
    # Предотвращаем создание дочерних процессов
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

from unsloth import FastLanguageModel
from datasets import load_from_disk
import torch
from trl import SFTTrainer
from transformers import TrainingArguments

import config

# Функция форматирования для датасета (определяем ДО if __name__)
def formatting_func(examples):
    # Unsloth требует список строк
    texts = []
    for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
        text = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        texts.append(text)
    return texts

# ===== ОСНОВНОЙ КОД (только при прямом запуске) =====
if __name__ == '__main__':
    # Загрузи модель с 4-bit для экономии VRAM
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_PATH,  # Из config.py
        max_seq_length=16384,  # Уменьшено с 32K до 16K для экономии VRAM (покрывает 95% историй)
        dtype=None,  # Auto-detect
        load_in_4bit=True,
        local_files_only=True,  # НЕ скачивать ничего онлайн, только локальные файлы
        device_map="auto"  # Автоматическое распределение по устройствам
    )
    
    # Добавь LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Rank LoRA, 16-32 хорошо
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )
    
    # Загрузи датасет
    dataset = load_from_disk("warhammer_dataset")
    
    # Тренировка
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,  # Используем функцию форматирования
        max_seq_length=16384,  # Совпадает с моделью (16K токенов)
        dataset_num_proc=None,  # Отключен мультипроцессинг - КРИТИЧНО для Windows!
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=1,  # Для контекста 16K токенов
            gradient_accumulation_steps=8,  # Эффективно = batch 8
            warmup_steps=5,
            max_steps=100,  # Для Warhammer датасета достаточно
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),  # Экономия VRAM
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",  # 8-bit оптимизатор для экономии памяти
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs"
        )
    )
    
    print("\n" + "="*60)
    print("Начинаю обучение...")
    print("="*60)
    print(f"Датасет: {len(dataset)} примеров")
    print(f"Max seq length: 16,384 токена (~45K символов)")
    print(f"Размер батча: 1 x 8 = 8 (effective)")
    print(f"Шагов обучения: 100 (~7.7 эпох)")
    print(f"Примерное время: 30-60 минут")
    print(f"VRAM использование: ~8-9GB")
    print(f"Мультипроцессинг: ВЫКЛЮЧЕН (стабильность)")
    print("="*60 + "\n")
    
    trainer.train()
    
    print("\n" + "="*60)
    print("[OK] Обучение завершено!")
    print(f"Сохраняю модель в: {config.FINETUNED_MODEL_PATH}")
    print("="*60)
    
    model.save_pretrained(config.FINETUNED_MODEL_PATH)
    tokenizer.save_pretrained(config.FINETUNED_MODEL_PATH)
    
    print("\n[SUCCESS] Модель успешно сохранена!")
    print(f"Путь: {config.FINETUNED_MODEL_PATH}/")
    print("="*60)