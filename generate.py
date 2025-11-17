from unsloth import FastLanguageModel
import torch
import config

# ===== ОПТИМИЗАЦИЯ ДЛЯ RTX 3060 TI 12GB =====
print("🔧 Загрузка модели для RTX 3060 Ti 12GB...")
print(f"📁 Путь: {config.FINETUNED_MODEL_PATH}")

model, tokenizer = FastLanguageModel.from_pretrained(
    config.FINETUNED_MODEL_PATH,
    max_seq_length=16384,  # ✅ Как при обучении (покрывает 82.7% историй)
    dtype=None,
    load_in_4bit=True      # ✅ Обязательно! Иначе не влезет в 12GB
)
FastLanguageModel.for_inference(model)

# Очистка VRAM перед генерацией
torch.cuda.empty_cache()

print(f"✅ Модель загружена. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB / 12GB")
print("")

prompt = """Write an epic Warhammer 40,000 story.

Theme: A battle between Space Marines and Orks on a forgotten planet.
Style: Epic, dramatic, with detailed combat scenes.
Length: Extended narrative, minimum 5000 words.

Story:"""
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

print("📝 Генерация истории Warhammer 40,000...")
print("⏱️  Ожидаемое время: 5-15 минут (зависит от длины)")
print(f"🎯 Максимум токенов: 12000 (~35-40K символов)")
print("")

outputs = model.generate(
    **inputs, 
    max_new_tokens=12000,   # ✅ ~35-40K символов (безопасно для 12GB)
    temperature=0.8,        # Креативность
    top_p=0.95,            # Разнообразие
    do_sample=True,
    use_cache=True         # ✅ Кешировать для скорости
)
story = tokenizer.decode(outputs[0])

# Очистка памяти после генерации
torch.cuda.empty_cache()

print("\n" + "="*60)
print("GENERATED STORY")
print("="*60 + "\n")
print(story)

# Сохранение в файл
with open("generated_story.txt", "w", encoding="utf-8") as f:
    f.write(story)

print("\n" + "="*60)
print(f"✅ История сохранена: generated_story.txt")
print(f"📊 Длина: {len(story)} символов")
print(f"💾 Использовано VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
print("="*60)