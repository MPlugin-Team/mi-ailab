# 🚀 Деплой: Mi-AiLab + Mi-AiPro в production

Mi-AiLab — это **инструмент тренировки**. Mi-AiPro — серверная часть для **раздачи ассистента пользователям**.

## Связка

```
Mi-AiLab (тренировка)  →  .pt модель или LoRA-адаптер
                                ↓
                          Конвертация в GGUF
                                ↓
                  Mi-AiPro (C# + LLamaSharp + Vulkan GPU)
                                ↓
                       REST API /api/v1/ask
                                ↓
                          Юзеры через UI/бота
```

## Этап 1 — Тренируй в Mi-AiLab

Через GUI (`🤖 Дообучение`) или CLI:

```bash
py cli.py lora-finetune \
    --model saiga_8b \
    --data data/texts/mi_ai_training_5000.txt \
    --output models/mi_ai_adapter
```

Получаешь `models/mi_ai_adapter/` (HuggingFace peft формат).

## Этап 2 — Конвертация в GGUF

См. [`scripts/convert_lora_to_gguf.md`](../scripts/convert_lora_to_gguf.md).

Кратко:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements/requirements-convert_lora_to_gguf.txt
py convert_lora_to_gguf.py \
    --base IlyaGusev/saiga_yandexgpt_8b \
    --outfile ../mi-aipro/data/mi_ai_adapter.gguf \
    ../mi-ailab/models/mi_ai_adapter
```

## Этап 3 — Запуск Mi-AiPro

```powershell
# Windows
$env:MI_AIPRO_GGUF = "data/saiga.gguf"
$env:MI_AIPRO_LORA = "data/mi_ai_adapter.gguf"
$env:MI_AIPRO_GPU_LAYERS = "35"
dotnet run --project Mi.AiPro.Server
```

Проверка:
```bash
curl http://localhost:5050/api/v1/info
```

## Варианты хостинга

| Где | Цена | Плюсы | Минусы |
|---|---|---|---|
| **Свой ПК + Cloudflare Tunnel** | $0 | Бесплатно, своя GPU | Зависит от твоего инета и ПК |
| **VPS RTX 3060** (Vast.ai / Contabo) | $30-50/мес | Стабильно | Слабая GPU |
| **VPS RTX 4090** | $200-400/мес | Быстро | Дорого |
| **A100 на час** (Vast.ai, RunPod) | $1-2/час | По требованию | Холодный старт ~5 мин |

## Доступ через интернет

### Вариант A — Cloudflare Tunnel (рекомендую)

Бесплатно, безопасно, свой домен.

```bash
# Установка
winget install cloudflare.cloudflared

# Логин
cloudflared tunnel login

# Создание туннеля
cloudflared tunnel create mi-ai

# Конфиг ~/.cloudflared/config.yml
tunnel: mi-ai
credentials-file: ~/.cloudflared/<uuid>.json
ingress:
  - hostname: mi-ai.yourdomain.com
    service: http://localhost:5050
  - service: http_status:404

# Запуск
cloudflared tunnel run mi-ai
```

Теперь `mi-ai.yourdomain.com` доступен из любой точки мира без открытия портов.

### Вариант B — ngrok (для теста)

```bash
ngrok http 5050
```

Получишь URL вроде `https://abc-123.ngrok.io`. Простое решение для пятиминутного теста.

### Вариант C — Port forwarding

Прокинь порт 5050 через роутер. Нужен белый IP. **Менее безопасно** — открыт всему интернету без CDN.

## Безопасность (обязательно перед публикацией)

1. **API-ключи**: добавь `Authorization: Bearer <token>` в Mi-AiPro middleware
2. **Rate limiting**: 10 запросов/мин на ключ
3. **Логирование**: сохраняй Q&A в JSONL для последующего fine-tune
4. **HTTPS**: обязательно (через Cloudflare автоматически)
5. **Без `MI_AIPRO_DEBUG`**: не раскрывай ошибки наружу

## Continuous improvement

```
Раз в неделю → собрать логи → отобрать хорошие Q&A
            ↓
Добавить в data/texts/mi_ai_training.txt
            ↓
Переобучить LoRA в Mi-AiLab
            ↓
Сконвертировать в GGUF, заменить адаптер
            ↓
Перезапустить Mi-AiPro
```

Так делают OpenAI и Anthropic — постепенное улучшение модели на реальных данных.
