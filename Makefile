.PHONY: install run test lint format check docker-build docker-run
install:
	uv sync
run:
	uv run streamlit run app.py
test:
	uv run pytest -q
lint:
	uv run ruff check .
format:
	uv run ruff format .
check:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest -W error -q
docker-build:
	docker build -t uao-neumonia .
docker-run:
	docker run --rm -p 8501:8501 uao-neumonia
