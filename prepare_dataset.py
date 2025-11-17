import json
import os
from datasets import Dataset
import config

# TXT folder with transcriptions
txt_folder = config.OUTPUT_DIR  # From config.py
data = []

print(f"Preparing dataset from: {txt_folder}")
print("Loading stories...")

for file in os.listdir(txt_folder):
    if file.endswith(".txt"):
        with open(os.path.join(txt_folder, file), "r", encoding="utf-8") as f:
            story = f.read().strip()
            # Split into prompt (beginning) and completion (continuation)
            if len(story) > 1000:  # Only long stories
                prompt = story[:500] + "\nContinue this Warhammer 40,000 story:"
                completion = story[500:]
                data.append({
                    "instruction": "Write a continuation of a Warhammer 40,000 story.",
                    "input": prompt,
                    "output": completion
                })

print(f"Total stories processed: {len(data)}")

# Save as JSONL
with open("dataset.jsonl", "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Saved: dataset.jsonl")

# Load into Hugging Face Dataset
dataset = Dataset.from_json("dataset.jsonl")
dataset.save_to_disk("warhammer_dataset")

print("Dataset saved to: warhammer_dataset/")
print(f"Ready for training with {len(dataset)} examples!")