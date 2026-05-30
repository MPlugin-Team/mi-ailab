# Конвертация LoRA-адаптера из peft в GGUF (для Mi-AiPro)

Mi-AiLab сохраняет LoRA-адаптеры в формате HuggingFace peft (`.safetensors`).
Mi-AiPro использует llama.cpp через LLamaSharp, который ждёт **GGUF**-формат.

Поэтому нужен один конвертационный шаг между «обучил адаптер» и
«подключил в Mi-AiPro».

## Шаг 1 — натренировать LoRA в Mi-AiLab

```bash
cd c:/Users/user/Desktop/mishatools/mi-ailab
py -3.11 cli.py lora-finetune \
    --model saiga_8b \
    --data data/texts/mi_ai_personality.txt \
    --epochs 5 \
    --output models/mi_ai_adapter
```

После этого в `models/mi_ai_adapter/` будут файлы:
- `adapter_config.json`
- `adapter_model.safetensors`

## Шаг 2 — клонировать llama.cpp (один раз)

```bash
cd c:/Users/user/Desktop/
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
py -3.11 -m pip install -r requirements/requirements-convert_lora_to_gguf.txt
```

## Шаг 3 — конвертировать в GGUF

```bash
cd c:/Users/user/Desktop/llama.cpp
py -3.11 convert_lora_to_gguf.py \
    --base IlyaGusev/saiga_yandexgpt_8b \
    --outfile c:/Users/user/Desktop/mishatools/mi-aipro/data/mi_ai_adapter.gguf \
    c:/Users/user/Desktop/mishatools/mi-ailab/models/mi_ai_adapter
```

Получишь `mi_ai_adapter.gguf` (~100-200 МБ).

## Шаг 4 — подключить в Mi-AiPro через ENV

```powershell
# Windows PowerShell
$env:MI_AIPRO_GGUF = "data/saiga.gguf"
$env:MI_AIPRO_LORA = "data/mi_ai_adapter.gguf"
$env:MI_AIPRO_LORA_SCALE = "1.0"
$env:MI_AIPRO_GPU_LAYERS = "35"

dotnet run --project Mi.AiPro.Server
```

```bash
# Linux/Mac
export MI_AIPRO_GGUF=data/saiga.gguf
export MI_AIPRO_LORA=data/mi_ai_adapter.gguf
export MI_AIPRO_LORA_SCALE=1.0

dotnet run --project Mi.AiPro.Server
```

## Шаг 5 — проверить

```bash
curl http://localhost:5050/api/v1/info
```

Должно вернуть:
```json
{
  "service": "Mi-AiPro",
  "version": "0.2.0",
  "model": "saiga + mi_ai_adapter.gguf",
  "base_model": "saiga",
  "adapters": ["mi_ai_adapter.gguf"],
  "display_name": "Mi-AI"
}
```

## Несколько адаптеров одновременно

Через `;` разделитель в `MI_AIPRO_LORA`:

```bash
export MI_AIPRO_LORA="data/mi_ai_personality.gguf;data/mi_ai_coding.gguf"
```

Все адаптеры применятся со scale из `MI_AIPRO_LORA_SCALE` (одинаковый для всех).

## Регулировка силы адаптера

`MI_AIPRO_LORA_SCALE=0.5` — половина эффекта (мягче).
`MI_AIPRO_LORA_SCALE=1.0` — полный эффект (стандарт).
`MI_AIPRO_LORA_SCALE=0` — выключен (как без адаптера).

Полезно если адаптер слишком сильно меняет стиль базовой Saiga.
