from unsloth import FastLanguageModel
import torch
import config

model, tokenizer = FastLanguageModel.from_pretrained(
    config.FINETUNED_MODEL_PATH,  # Из config.py
    max_seq_length=8192,
    dtype=None,
    load_in_4bit=True
)
FastLanguageModel.for_inference(model)  # Ускорение

prompt = """Write an epic Warhammer 40,000 story.

Theme: A battle between Space Marines and Orks on a forgotten planet.
Style: Epic, dramatic, with detailed combat scenes.
Length: Extended narrative, minimum 5000 words.

Story:"""
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

print("Generating Warhammer 40,000 story...")
print("This may take 5-10 minutes...\n")

outputs = model.generate(**inputs, max_new_tokens=20000,  # For 50k+ characters
                         temperature=0.8, top_p=0.95, do_sample=True)
story = tokenizer.decode(outputs[0])

print("\n" + "="*60)
print("GENERATED STORY")
print("="*60 + "\n")
print(story)

# Optionally save to file
with open("generated_story.txt", "w", encoding="utf-8") as f:
    f.write(story)

print("\n" + "="*60)
print("Story saved to: generated_story.txt")
print("="*60)