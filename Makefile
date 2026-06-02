.PHONY: collect features train simulate backend frontend

collect:
	python ml/collect/collect_all.py

features:
	python ml/features.py

train:
	python ml/train.py

simulate:
	python ml/simulate.py

explain:
	python ml/explain.py

backend:
	uvicorn backend.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

setup:
	pip install -r requirements.txt
	cd frontend && npm install

all: collect features train simulate
