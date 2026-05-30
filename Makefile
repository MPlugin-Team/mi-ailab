# Mi-AiLab — короткие команды для частых задач.
# Использование: make <команда>

.PHONY: help install gui web cli train gen data docker test clean

help:
	@echo "Mi-AiLab — доступные команды:"
	@echo ""
	@echo "  make install   — установить зависимости"
	@echo "  make gui       — запустить GUI"
	@echo "  make web       — запустить web-режим (порт 8550)"
	@echo "  make cli       — показать CLI справку"
	@echo "  make train     — тренировка по умолчанию (Sherlock Transformer)"
	@echo "  make gen       — перегенерить mi_ai_training_5000.txt"
	@echo "  make data      — собрать все Q&A датасеты"
	@echo "  make docker    — собрать Docker-образ"
	@echo "  make test      — проверка импортов всех модулей"
	@echo "  make clean     — удалить кэш Python"

install:
	pip install -r requirements.txt

gui:
	python app.py

web:
	python app.py --web --host 0.0.0.0 --port 8550

cli:
	python cli.py --help

train:
	python cli.py train --config docs/examples/transformer_sherlock.yaml

gen:
	python mi.py

data:
	python mi.py
	python scripts/generate_vpn_qa.py
	python scripts/combine_datasets.py

docker:
	docker build -t mi-ailab .
	@echo "Запуск: docker run -p 8550:8550 mi-ailab"

test:
	python -c "from src import theme, datasets, neural_net, text_datasets, text_model, hardware, model_storage, cnn_model, image_datasets, bpe_tokenizer, streaming_dataset, transformer_model, instruction_datasets, lora_finetune, tooltips; print('All imports OK')"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."
