# Qwen 2.5 7B Fine-tuning для Warhammer 40K

Дообучение Qwen 2.5 7B на историях Warhammer 40K (104 истории, 16K контекст, LoRA)

## Быстрый старт на RunPod

```bash
cd /workspace
git clone https://github.com/sayonaraman/warhammer-qwen-finetuning.git
cd warhammer-qwen-finetuning
chmod +x runpod_quickstart.sh
./runpod_quickstart.sh
```

**Стоимость:** ~$0.43 (RTX 3090, 75 минут)

## Что внутри

- `runpod_quickstart.sh` - полная автоматизация
- `train_runpod.py` - обучение для RunPod
- `prepare_dataset.py` - создание датасета
- `config.py` - настройки

## После обучения

```bash
zip -r fine_tuned_model.zip fine_tuned_model/
```

⚠️ **Остановите Pod после работы!**

