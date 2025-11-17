"""
Qwen 2.5 7B Fine-tuning для Warhammer 40K
Оптимизировано для RunPod (RTX 3090/4090)
"""

from unsloth import FastLanguageModel
from datasets import load_from_disk
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
import config

print("\n" + "="*60)
print("🦥 Unsloth + Qwen 2.5 7B Fine-tuning")
print("="*60)

# Функция форматирования датасета
def formatting_func(examples):
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

print("\n[1/6] Загрузка модели Qwen 2.5 7B...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.MODEL_PATH,
    max_seq_length=16384,  # 16K контекст (покрывает 82.7% историй)
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # 4-bit квантизация для экономии VRAM
)

print("\n[2/6] Добавление LoRA адаптеров...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None
)

print("\n[3/6] Загрузка датасета...")
dataset = load_from_disk("warhammer_dataset")
print(f"   ✅ Загружено {len(dataset)} примеров")
print(f"   📊 Средний размер: ~60K символов/история")

print("\n[4/6] Настройка тренера...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_func,
    max_seq_length=16384,
    dataset_num_proc=2,  # RunPod поддерживает multiprocessing
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,  # Для 24GB VRAM
        gradient_accumulation_steps=4,  # Эффективный batch = 8
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs"
    )
)

print("\n[5/6] Начинаю обучение...")
print("="*60)
print(f"📊 Параметры:")
print(f"   • Датасет: {len(dataset)} примеров")
print(f"   • Контекст: 16,384 токена (~45K символов)")
print(f"   • Batch size: 2 × 4 = 8 (effective)")
print(f"   • Шагов: 100 (~7.7 эпох)")
print(f"   • VRAM: ~14-16GB")
print(f"   • Время: ~45-90 минут")
print(f"   • Стоимость: ~$0.26-0.52 (RTX 3090)")
print("="*60 + "\n")

# ОБУЧЕНИЕ
trainer.train()

print("\n[6/6] Сохранение модели...")
model.save_pretrained(config.FINETUNED_MODEL_PATH)
tokenizer.save_pretrained(config.FINETUNED_MODEL_PATH)

print("\n" + "="*60)
print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print("="*60)
print(f"📁 Модель сохранена: {config.FINETUNED_MODEL_PATH}/")
print("\n💾 Скачайте папку с моделью через RunPod interface")
print("   или используйте: zip -r fine_tuned_model.zip fine_tuned_model/")
print("="*60)

