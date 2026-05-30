# Dockerfile для Mi-AiLab — образ с готовым окружением.
# Сборка: docker build -t mi-ailab .
# Запуск GUI веб-режима: docker run -p 8550:8550 mi-ailab
# Запуск CLI: docker run -v $(pwd)/models:/app/models mi-ailab python cli.py hardware

FROM python:3.11-slim

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Сначала только requirements — Docker закэширует слой
COPY requirements.txt .

# CPU-версия torch для образа (для GPU нужен nvidia/cuda базовый образ)
RUN pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Остальной код
COPY . .

# Веб-режим по умолчанию
EXPOSE 8550
CMD ["python", "app.py", "--web", "--host", "0.0.0.0", "--port", "8550"]
